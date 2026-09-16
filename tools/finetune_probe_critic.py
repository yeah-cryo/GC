import argparse
import hashlib
import json
import os
import shutil
import time
import uuid
from importlib.metadata import version
from itertools import islice
from pathlib import Path

import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader

from aigi_detection.datasets import ImageDataset, TrainTransform, ValidationTransform, read_manifest
from aigi_detection.datasets.genimage import LABELS, collate_readable
from aigi_detection.datasets.probe import build_probe_records, write_probe_manifest
from aigi_detection.engine import atomic_save, rng_state, seed_everything, seed_worker, validate, write_json
from aigi_detection.models.backbones import build_critic


def file_sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description='PROBE-style ResNet-50 detector adaptation.')
    parser.add_argument('--config', default='configs/experiments/resnet50_probe_finetune.yaml')
    parser.add_argument('--resume')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if config['precision'] != 'bf16' or config['pretrain_batch_size'] != config['probe_batch_size']:
        raise ValueError('Expected BF16 and equal-size original/PROBE batches.')
    if config['probe_loss_weight'] != 0.5:
        raise ValueError('PROBE fine-tuning requires w=0.5.')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('A BF16 CUDA GPU is required.')
    device = torch.device('cuda')
    seed_everything(config['seed'])
    torch.set_num_threads(2)
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and any((output / name).exists() for name in ('latest.pt', 'best.pt', 'metrics.jsonl')):
        raise FileExistsError('Existing fine-tuning run found; pass --resume latest.pt.')

    source_snapshot = output / 'source_critic.pt'
    if not source_snapshot.exists():
        shutil.copyfile(config['source_checkpoint'], source_snapshot.with_suffix('.tmp'))
        os.replace(source_snapshot.with_suffix('.tmp'), source_snapshot)
    source_hash = file_sha256(source_snapshot)
    source = torch.load(source_snapshot, map_location='cpu', weights_only=False)
    if source['label_mapping'] != LABELS:
        raise ValueError('Expected the source critic to use nature=0 and ai=1.')
    source_epoch = source.get('epoch')

    manifests = output / 'manifests'
    probe_records = build_probe_records(config['generated_metrics'], config['generated_root'],
                                        config['coco_images_root'], config['probe_fake_images'],
                                        config.get('probe_fake_source', 'guided_sd14'))
    probe_hash = write_probe_manifest(probe_records, manifests / 'probe.jsonl')
    pretrain_records = read_manifest(config['pretrain_manifest'])
    validation_records = read_manifest(config['validation_manifest'])
    pretrain_hash = hashlib.sha256(Path(config['pretrain_manifest']).read_bytes()).hexdigest()
    validation_hash = hashlib.sha256(Path(config['validation_manifest']).read_bytes()).hexdigest()
    manifest_metadata = {
        'policy': ('Additional data contains one caption-linked COCO real and one '
                   f"{config.get('probe_fake_source', 'guided_sd14')} fake per prompt"),
        'probe_records': len(probe_records), 'probe_real': config['probe_fake_images'],
        'probe_fake': config['probe_fake_images'],
        'unique_coco_images': len({record['image_id'] for record in probe_records if record['label'] == 0}),
        'sha256': {'probe': probe_hash, 'pretrain': pretrain_hash, 'validation': validation_hash},
    }
    write_json(manifest_metadata, manifests / 'manifest_identity.json')

    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    is_resume = checkpoint is not None
    if checkpoint:
        if checkpoint['label_mapping'] != LABELS or checkpoint['manifest_sha256'] != manifest_metadata['sha256']:
            raise ValueError('Resume labels or manifests differ.')
        if checkpoint['configuration'] != config or checkpoint['source_checkpoint_sha256'] != source_hash:
            raise ValueError('Resume configuration or source critic differs.')
    model = build_critic().to(device)
    model.load_state_dict(checkpoint['model'] if checkpoint else source['model'], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                  weight_decay=config['weight_decay'],
                                  betas=tuple(config['betas']), eps=config['epsilon'])
    start_epoch, best, stale = 1, float('-inf'), 0
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best, stale = checkpoint['best_score'], checkpoint['early_stopping_counter']
    del source, checkpoint

    transform = TrainTransform(config['crop_size'])
    pretrain_set = ImageDataset(config['pretrain_root'], pretrain_records, transform,
                                corruption_policy='replace', error_log=output / 'corrupt_pretrain.jsonl')
    probe_set = ImageDataset('/', probe_records, transform, corruption_policy='replace',
                             error_log=output / 'corrupt_probe.jsonl')
    validation_set = ImageDataset(config['pretrain_root'], validation_records,
                                  ValidationTransform(config['crop_size']), corruption_policy='skip',
                                  error_log=output / 'corrupt_validation.jsonl')
    common = {'num_workers': config['workers_per_loader'], 'pin_memory': True,
              'worker_init_fn': seed_worker, 'persistent_workers': False}
    validation_loader = DataLoader(validation_set, batch_size=config['validation_batch_size'],
                                   shuffle=False, collate_fn=collate_readable, **common)
    steps_per_epoch = len(probe_set) // config['probe_batch_size']
    if steps_per_epoch * config['probe_batch_size'] != len(probe_set):
        raise ValueError('D_PROBE must divide evenly into full batches.')

    (output / 'configuration.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    write_json({'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                'packages': {name: version(name) for name in ('torchvision', 'numpy', 'Pillow', 'PyYAML')},
                'gpu': torch.cuda.get_device_name(), 'label_mapping': LABELS}, output / 'environment.json')
    write_json({'source': str(Path(config['source_checkpoint']).resolve()), 'copied_to': str(source_snapshot),
                'sha256': source_hash, 'epoch': source_epoch}, output / 'source_critic_identity.json')
    if not is_resume and (output / 'metrics.jsonl').exists():
        raise FileExistsError('Metrics exist without an explicit resume checkpoint.')
    if args.resume and (output / 'metrics.jsonl').exists():
        rows = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines() if line]
        (output / 'metrics.jsonl').write_text(
            ''.join(json.dumps(row) + '\n' for row in rows if row['epoch'] < start_epoch))

    tracker = None
    if config.get('swanlab'):
        import swanlab
        tracking_path = output / 'swanlab_run.json'
        tracking = json.loads(tracking_path.read_text()) if tracking_path.exists() else {'id': uuid.uuid4().hex[:16]}
        tracker = swanlab.init(project=config['swanlab']['project'],
                               experiment_name=config['swanlab']['experiment_name'],
                               config=config, mode='cloud', public=False,
                               logdir=str(output / 'swanlab'), id=tracking['id'], resume='allow')
        tracking['url'] = tracker.url
        write_json(tracking, tracking_path)

    if stale >= config['patience']:
        print('Checkpoint has already reached early stopping.', flush=True)
        if tracker:
            swanlab.finish()
        return
    for epoch in range(start_epoch, config['max_epochs'] + 1):
        began = time.monotonic()
        pretrain_generator = torch.Generator().manual_seed(config['seed'] + 2 * epoch)
        probe_generator = torch.Generator().manual_seed(config['seed'] + 2 * epoch + 1)
        pretrain_loader = DataLoader(pretrain_set, batch_size=config['pretrain_batch_size'],
                                     shuffle=True, drop_last=True, generator=pretrain_generator, **common)
        probe_loader = DataLoader(probe_set, batch_size=config['probe_batch_size'],
                                  shuffle=True, drop_last=True, generator=probe_generator, **common)
        model.train()
        totals = torch.zeros(3, dtype=torch.float64, device=device)
        pretrain_iterator = iter(pretrain_loader)
        for step, (probe_images, probe_labels) in enumerate(probe_loader, start=1):
            pretrain_images, pretrain_labels = next(pretrain_iterator)
            pretrain_images = pretrain_images.to(device, non_blocking=True)
            pretrain_labels = pretrain_labels.to(device, non_blocking=True).float()
            probe_images = probe_images.to(device, non_blocking=True)
            probe_labels = probe_labels.to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pretrain_loss = F.binary_cross_entropy_with_logits(
                    model(pretrain_images).flatten(), pretrain_labels)
                (0.5 * pretrain_loss).backward()
                probe_loss = F.binary_cross_entropy_with_logits(model(probe_images).flatten(), probe_labels)
                (0.5 * probe_loss).backward()
            optimizer.step()
            totals += torch.tensor([pretrain_loss.item(), probe_loss.item(), 1.0],
                                   dtype=torch.float64, device=device)
            if step == 1 or step % config['log_every_steps'] == 0:
                elapsed = time.monotonic() - began
                weighted = 0.5 * (pretrain_loss.item() + probe_loss.item())
                print(f'epoch={epoch} step={step}/{steps_per_epoch} loss={weighted:.5f} '
                      f'pre={pretrain_loss.item():.5f} probe={probe_loss.item():.5f} '
                      f'images_per_second={2 * step * config["probe_batch_size"] / elapsed:.1f}', flush=True)
                if tracker:
                    global_step = (epoch - 1) * steps_per_epoch + step
                    tracker.log({'train/loss': weighted, 'train/pretrain_loss': pretrain_loss.item(),
                                 'train/probe_loss': probe_loss.item(), 'train/epoch': epoch,
                                 'train/learning_rate': optimizer.param_groups[0]['lr']}, step=global_step)
        validation = validate(model, validation_loader, device)
        improved = validation['balanced_accuracy'] > best + config['improvement_delta']
        best, stale = (validation['balanced_accuracy'], 0) if improved else (best, stale + 1)
        pretrain_mean = (totals[0] / totals[2]).item()
        probe_mean = (totals[1] / totals[2]).item()
        row = {'epoch': epoch, 'steps': steps_per_epoch,
               'pretrain_images_seen': steps_per_epoch * config['pretrain_batch_size'],
               'probe_images_seen': steps_per_epoch * config['probe_batch_size'],
               'pretrain_loss': pretrain_mean, 'probe_loss': probe_mean,
               'train_loss': 0.5 * (pretrain_mean + probe_mean), 'validation': validation,
               'best_score': best, 'early_stopping_counter': stale, 'seconds': time.monotonic() - began}
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                 'best_score': best, 'early_stopping_counter': stale, 'configuration': config,
                 'label_mapping': LABELS, 'manifest_sha256': manifest_metadata['sha256'],
                 'source_checkpoint_sha256': source_hash, 'rng_state': rng_state(), 'metrics': row,
                 'architecture': 'resnet50', 'output_semantics': 'one logit; sigmoid = P(fake)'}
        if improved:
            atomic_save(state, output / 'best.pt')
        atomic_save(state, output / 'latest.pt')
        with (output / 'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
        if tracker:
            tracker.log({**{f'validation/{key}': value for key, value in validation.items()},
                         'train/epoch_loss': row['train_loss'], 'validation/best_score': best,
                         'validation/early_stopping_counter': stale}, step=epoch * steps_per_epoch)
        if stale >= config['patience']:
            print(f'Early stopping after epoch {epoch}.', flush=True)
            break
    if tracker:
        swanlab.finish()


if __name__ == '__main__':
    main()
