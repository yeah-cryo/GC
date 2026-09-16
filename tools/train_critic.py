import argparse
import json
import os
import time
import uuid
from importlib.metadata import version
from pathlib import Path
from itertools import islice
from contextlib import nullcontext

import torch
import torch.distributed as dist
import yaml
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

from aigi_detection.datasets import ImageDataset, TrainTransform, ValidationTransform, build_manifests, read_manifest
from aigi_detection.datasets.genimage import LABELS, collate_readable
from aigi_detection.engine import atomic_save, restore_rng, rng_state, seed_everything, seed_worker, validate, write_json
from aigi_detection.models.backbones import build_critic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/experiments/resnet50_critic.yaml')
    parser.add_argument('--resume')
    parser.add_argument('--data-root')
    parser.add_argument('--pretrained')
    parser.add_argument('--output')
    args = parser.parse_args()
    with open(args.config) as handle:
        config = yaml.safe_load(handle)
    for key in ('data_root', 'pretrained', 'output'):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    rank, local_rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('LOCAL_RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    if world != config['world_size']:
        raise RuntimeError('Number of launched ranks must match configuration world_size.')
    if torch.cuda.device_count() < world:
        raise RuntimeError(f'{world} visible CUDA GPU(s) required.')
    accumulation = config['gradient_accumulation_steps']
    if accumulation < 1 or world * config['batch_size_per_gpu'] * accumulation != 128:
        raise ValueError('Effective global batch size must be 128.')
    if not Path(config['data_root']).joinpath('train').is_dir():
        raise FileNotFoundError(f"Official training split missing: {config['data_root']}/train")
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 support is required.')
    if config['precision'] != 'bf16' or config['batch_size_per_gpu'] != 64:
        raise ValueError('This experiment requires BF16 and batch size 64 per GPU.')
    if world > 1:
        dist.init_process_group('nccl')
    try:
        run(args, config, rank, world, device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run(args, config, rank, world, device):
    seed_everything(config['seed'] + rank)
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        if not args.resume and any((output / name).exists() for name in ('latest.pt', 'best.pt', 'metrics.jsonl')):
            raise FileExistsError('Existing run found: use --resume or a new --output directory.')
        print('Preparing deterministic train/internal-validation manifests...', flush=True)
        build_manifests(config['data_root'], output / 'manifests', config['seed'], config['validation_fraction'], config['expected_per_class'])
        print('Manifests ready: 307800 train / 16200 internal validation.', flush=True)
    if dist.is_initialized():
        dist.barrier()
    split = json.loads((output / 'manifests/split.json').read_text())
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    if checkpoint:
        if checkpoint['label_mapping'] != LABELS or checkpoint['split_sha256'] != split['sha256']:
            raise ValueError('Checkpoint labels or split manifests differ.')
        for key, value in config.items():
            if key not in ('output', 'pretrained', 'data_root') and checkpoint['configuration'][key] != value:
                raise ValueError(f'Resume configuration mismatch: {key}')
    model = build_critic(None if checkpoint else config['pretrained']).to(device)
    if checkpoint:
        model.load_state_dict(checkpoint['model'], strict=True)
    if world > 1:
        model = DDP(model, device_ids=[device.index], broadcast_buffers=True)
    raw_model = model.module if world > 1 else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'],
                                 betas=tuple(config['betas']), eps=config['epsilon'])
    start, best, stale = 1, float('-inf'), 0
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        start, best, stale = checkpoint['epoch'] + 1, checkpoint['best_score'], checkpoint['early_stopping_counter']
    train_set = ImageDataset(config['data_root'], read_manifest(output / 'manifests/train.jsonl'), TrainTransform(config['crop_size']),
                             corruption_policy='replace', error_log=output / 'corrupt_train.jsonl')
    val_set = ImageDataset(config['data_root'], read_manifest(output / 'manifests/validation.jsonl'), ValidationTransform(config['crop_size']),
                           corruption_policy='skip', error_log=output / 'corrupt_validation.jsonl')
    sampler = DistributedSampler(train_set, world, rank, shuffle=True, seed=config['seed'], drop_last=True)
    generator = torch.Generator()
    common = {'batch_size': config['batch_size_per_gpu'], 'num_workers': config['workers_per_gpu'],
              'pin_memory': True, 'worker_init_fn': seed_worker, 'persistent_workers': False}
    train_loader = DataLoader(train_set, sampler=sampler, drop_last=True, generator=generator, **common)
    # Striding partitions validation without DistributedSampler's duplicate padding.
    val_loader = DataLoader(Subset(val_set, range(rank, len(val_set), world)), shuffle=False, collate_fn=collate_readable, **common)
    if checkpoint:
        restore_rng(checkpoint['rng_states'][rank])
    if rank == 0:
        (output / 'configuration.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        write_json({'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                    'packages': {name: version(name) for name in ('torchvision', 'numpy', 'Pillow', 'PyYAML')},
                    'gpus': [torch.cuda.get_device_name(i) for i in range(world)], 'label_mapping': LABELS}, output / 'environment.json')
        # Remove metrics from a partially committed epoch when resuming latest.pt.
        metrics_file = output / 'metrics.jsonl'
        if checkpoint and metrics_file.exists():
            rows = [json.loads(line) for line in metrics_file.read_text().splitlines() if line.strip()]
            metrics_file.write_text(''.join(json.dumps(row) + '\n' for row in rows if row['epoch'] < start))
    if stale >= config['patience']:
        if rank == 0:
            print('Checkpoint has already reached early stopping.', flush=True)
        return
    tracker = None
    if rank == 0 and config.get('swanlab'):
        import swanlab
        tracking_path = output / 'swanlab_run.json'
        tracking = json.loads(tracking_path.read_text()) if tracking_path.exists() else {'id': uuid.uuid4().hex[:16]}
        tracker = swanlab.init(project=config['swanlab']['project'],
                               experiment_name=config['swanlab']['experiment_name'],
                               config=config, mode='cloud', public=False,
                               logdir=str(output / 'swanlab'), id=tracking['id'], resume='allow')
        tracking['url'] = tracker.url
        write_json(tracking, tracking_path)
    for epoch in range(start, config['max_epochs'] + 1):
        began = time.monotonic()
        sampler.set_epoch(epoch)
        generator.manual_seed(config['seed'] + epoch * world + rank)
        model.train()
        totals = torch.zeros(2, dtype=torch.float64, device=device)
        accumulation = config['gradient_accumulation_steps']
        micro_steps = len(train_loader) // accumulation * accumulation
        if not micro_steps:
            raise ValueError('Training split is too small for one effective batch.')
        if rank == 0:
            print(f'Starting epoch {epoch}: {micro_steps // accumulation} optimizer steps, effective batch=128', flush=True)
        optimizer.zero_grad(set_to_none=True)
        for step, (images, labels) in enumerate(islice(train_loader, micro_steps), 1):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True).float()
            update = step % accumulation == 0
            context = model.no_sync() if world > 1 and not update else nullcontext()
            with context:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(images).flatten()
                    loss = F.binary_cross_entropy_with_logits(logits, labels)
                (loss / accumulation).backward()
            if update:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            totals[0] += loss.detach().double() * labels.numel()
            totals[1] += labels.numel()
            if rank == 0 and (step == accumulation or step % 100 == 0):
                elapsed = time.monotonic() - began
                print(f'epoch={epoch} micro_step={step}/{micro_steps} loss={loss.item():.5f} images_per_second={step * world * config["batch_size_per_gpu"] / elapsed:.1f}', flush=True)
                if tracker:
                    tracker.log({'train/loss': loss.item(), 'train/epoch': epoch,
                                 'train/images_per_second': step * world * config['batch_size_per_gpu'] / elapsed,
                                 'train/learning_rate': optimizer.param_groups[0]['lr']},
                                step=(epoch - 1) * (micro_steps // accumulation) + step // accumulation)
        if dist.is_initialized():
            dist.all_reduce(totals)
        metrics = validate(raw_model, val_loader, device)
        improved = metrics['balanced_accuracy'] > best + config['improvement_delta']
        best, stale = (metrics['balanced_accuracy'], 0) if improved else (best, stale + 1)
        states = [None] * world
        if dist.is_initialized():
            dist.all_gather_object(states, rng_state())
        else:
            states[0] = rng_state()
        if rank == 0:
            row = {'epoch': epoch, 'train_loss': (totals[0] / totals[1]).item(), 'train_images_seen': int(totals[1].item()),
                   'validation': metrics, 'best_score': best, 'early_stopping_counter': stale, 'seconds': time.monotonic() - began}
            state = {'model': raw_model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                     'best_score': best, 'early_stopping_counter': stale, 'configuration': config, 'label_mapping': LABELS,
                     'split_sha256': split['sha256'], 'rng_states': states, 'metrics': row,
                     'architecture': 'resnet50', 'output_semantics': 'one logit; sigmoid = P(fake)'}
            if improved:
                atomic_save(state, output / 'best.pt')
            atomic_save(state, output / 'latest.pt')
            with (output / 'metrics.jsonl').open('a') as handle:
                handle.write(json.dumps(row) + '\n')
            print(json.dumps(row), flush=True)
            if tracker:
                tracker.log({**{f'validation/{key}': value for key, value in metrics.items()},
                             'train/epoch_loss': row['train_loss'], 'validation/best_score': best,
                             'validation/early_stopping_counter': stale, 'train/epoch_seconds': row['seconds']},
                            step=epoch * (micro_steps // accumulation))
        if dist.is_initialized():
            dist.barrier()
        if stale >= config['patience']:
            break
    if tracker:
        swanlab.finish()


if __name__ == '__main__':
    main()
