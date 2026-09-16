import argparse
import hashlib
import json
import math
import os
import shutil
import time
import uuid
from importlib.metadata import version
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.nn import functional as F

from aigi_detection.datasets import TrainTransform
from aigi_detection.datasets.coco_prompts import read_prompt_manifest
from aigi_detection.engine import atomic_save, restore_rng, rng_state, seed_everything, write_json
from aigi_detection.models.backbones import build_critic


LABELS = {'nature': 0, 'ai': 1}


def file_sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def real_path(root, image_id):
    return Path(root) / f'COCO_train2014_{int(image_id):012d}.jpg'


def fake_path(root, index):
    return Path(root) / f'fake_{index:05d}.png'


def load_augmented(path, transform, label):
    with Image.open(path) as image:
        return transform(image.convert('RGB'), label)


def save_state(path, model, optimizer, scheduler, completed, steps, config,
               data_identity, history):
    state = {
        'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(), 'completed_samples': completed,
        'optimizer_steps': steps, 'configuration': config,
        'label_mapping': LABELS, 'data_identity': data_identity,
        'rng_state': rng_state(), 'metrics': history[-1] if history else None,
        'history': history, 'architecture': 'resnet50',
        'output_semantics': 'one logit; sigmoid = P(fake)',
    }
    atomic_save(state, path)


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune ImageNet ResNet-50 on saved online-guided pairs in original order.')
    parser.add_argument('--config', default='configs/experiments/resnet50_saved_guided_cosine_coco20k.yaml')
    parser.add_argument('--resume', nargs='?', const='auto')
    parser.add_argument('--max-samples', type=int,
                        help='Development aid: stop after this many total pairs.')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('A BF16 CUDA GPU is required.')
    if config['precision'] != 'bf16' or config['scheduler'] != 'cosine':
        raise ValueError('This experiment requires BF16 and cosine learning-rate decay.')
    if config['pair_batch_size'] < 1 or config['micro_batch_size'] < 1:
        raise ValueError('Batch sizes must be positive.')
    if not 0 <= config['minimum_learning_rate'] < config['learning_rate']:
        raise ValueError('minimum_learning_rate must be in [0, learning_rate).')
    sampling_order = config.get('sampling_order', 'sequential')
    if sampling_order not in ('sequential', 'random_permutation'):
        raise ValueError('sampling_order must be sequential or random_permutation.')

    seed_everything(config['seed'])
    torch.set_num_threads(2)
    device = torch.device('cuda')
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    latest_path = output / 'latest.pt'
    if args.resume == 'auto':
        args.resume = str(latest_path)
    if not args.resume and latest_path.exists():
        raise FileExistsError(f'Existing run found at {output}; pass --resume.')

    prompt_source = Path(config['prompt_manifest'])
    generation_source = Path(config['generation_metrics'])
    prompt_records = read_prompt_manifest(prompt_source, config['num_images'])
    generation_records = read_prompt_manifest(generation_source, config['num_images'])
    expected_indices = list(range(config['num_images']))
    if (len(prompt_records) != config['num_images'] or
            [row['prompt_index'] for row in prompt_records] != expected_indices):
        raise ValueError('Prompt manifest must contain consecutive indices for all images.')
    if (len(generation_records) != config['num_images'] or
            [row['prompt_index'] for row in generation_records] != expected_indices):
        raise ValueError('Generation metrics must contain consecutive indices for all images.')
    if any(int(prompt['image_id']) != int(generated['image_id'])
           for prompt, generated in zip(prompt_records, generation_records)):
        raise ValueError('Prompt and generated-image records disagree on COCO image IDs.')
    image_ids = [int(row['image_id']) for row in prompt_records]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError('Expected exactly one record per unique COCO image ID.')
    for key in ('coco_images_root', 'generated_images_root'):
        if not Path(config[key]).is_dir():
            raise FileNotFoundError(f'{key} is missing: {config[key]}')
    if not fake_path(config['generated_images_root'], 0).is_file() or not fake_path(
            config['generated_images_root'], config['num_images'] - 1).is_file():
        raise FileNotFoundError('The saved generated-image sequence is incomplete at an endpoint.')

    if sampling_order == 'sequential':
        sample_indices = expected_indices
        sampling_seed = None
    else:
        sampling_seed = int(config['sampling_seed'])
        order_generator = torch.Generator().manual_seed(sampling_seed)
        sample_indices = torch.randperm(config['num_images'], generator=order_generator).tolist()
    order_payload = {'sampling_order': sampling_order, 'sampling_seed': sampling_seed,
                     'indices': sample_indices}
    order_text = json.dumps(order_payload, separators=(',', ':')) + '\n'
    order_hash = hashlib.sha256(order_text.encode()).hexdigest()

    data_identity = {
        'prompt_manifest': str(prompt_source.resolve()),
        'prompt_manifest_sha256': file_sha256(prompt_source),
        'generation_metrics': str(generation_source.resolve()),
        'generation_metrics_sha256': file_sha256(generation_source),
        'generated_images_root': str(Path(config['generated_images_root']).resolve()),
        'records': config['num_images'], 'unique_image_ids': len(set(image_ids)),
        'sampling_order': sampling_order, 'sampling_seed': sampling_seed,
        'sample_order_sha256': order_hash,
    }
    prompt_snapshot = output / 'prompts.jsonl'
    generation_snapshot = output / 'source_generation_metrics.jsonl'
    for source, snapshot, digest_key in (
            (prompt_source, prompt_snapshot, 'prompt_manifest_sha256'),
            (generation_source, generation_snapshot, 'generation_metrics_sha256')):
        if snapshot.exists() and file_sha256(snapshot) != data_identity[digest_key]:
            raise ValueError(f'Saved source snapshot differs: {snapshot}')
        if not snapshot.exists():
            shutil.copyfile(source, snapshot)

    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    if checkpoint:
        if checkpoint['configuration'] != config or checkpoint['data_identity'] != data_identity:
            raise ValueError('Resume configuration or data identity differs.')
        if checkpoint['label_mapping'] != LABELS:
            raise ValueError('Resume label mapping differs.')
    model = build_critic(None if checkpoint else config['pretrained']).to(device)
    if checkpoint:
        model.load_state_dict(checkpoint['model'], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                  weight_decay=config['weight_decay'], betas=tuple(config['betas']),
                                  eps=config['epsilon'])
    total_steps = math.ceil(config['num_images'] / config['pair_batch_size'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=config['minimum_learning_rate'])
    completed, optimizer_steps, history = 0, 0, []
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        completed = int(checkpoint['completed_samples'])
        optimizer_steps = int(checkpoint['optimizer_steps'])
        history = checkpoint.get('history', [])
        restore_rng(checkpoint['rng_state'])
    del checkpoint

    metrics_path = output / 'metrics.jsonl'
    metrics_path.write_text(''.join(json.dumps(row) + '\n' for row in history), encoding='utf-8')
    (output / 'configuration.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    write_json(data_identity, output / 'data_identity.json')
    order_path = output / 'sample_order.json'
    if order_path.exists() and order_path.read_text(encoding='utf-8') != order_text:
        raise ValueError('Saved sample order differs from the reconstructed order.')
    order_path.write_text(order_text, encoding='utf-8')
    write_json({'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                'packages': {name: version(name) for name in ('torchvision', 'numpy', 'Pillow',
                                                               'PyYAML', 'swanlab')},
                'gpu': torch.cuda.get_device_name(), 'label_mapping': LABELS},
               output / 'environment.json')
    if not latest_path.exists():
        save_state(latest_path, model, optimizer, scheduler, completed, optimizer_steps,
                   config, data_identity, history)

    tracker = None
    if config.get('swanlab'):
        # Tracking must not change augmentation or within-block shuffle order.
        tracker_rng_state = rng_state()
        import swanlab
        tracking_path = output / 'swanlab_run.json'
        tracking = json.loads(tracking_path.read_text()) if tracking_path.exists() else {'id': uuid.uuid4().hex[:16]}
        tracker = swanlab.init(project=config['swanlab']['project'],
                               experiment_name=config['swanlab']['experiment_name'],
                               config=config, mode='cloud', public=False,
                               logdir=str(output / 'swanlab'), id=tracking['id'], resume='allow')
        tracking['url'] = tracker.url
        write_json(tracking, tracking_path)
        restore_rng(tracker_rng_state)

    transform = TrainTransform(config['crop_size'])
    total = min(config['num_images'], args.max_samples or config['num_images'])
    started = time.monotonic()
    model.train()
    while completed < total:
        block_start = completed
        block_end = min(total, block_start + config['pair_batch_size'])
        block_indices = sample_indices[block_start:block_end]
        real_images, fake_images = [], []
        for index in block_indices:
            record = prompt_records[index]
            real_images.append(load_augmented(real_path(config['coco_images_root'], record['image_id']),
                                              transform, 0))
            fake_images.append(load_augmented(fake_path(config['generated_images_root'], index),
                                              transform, 1))
        images = torch.stack(real_images + fake_images)
        labels = torch.cat((torch.zeros(len(real_images)), torch.ones(len(fake_images))))
        order = torch.randperm(len(labels))
        images, labels = images[order], labels[order]
        learning_rate = optimizer.param_groups[0]['lr']
        optimizer.zero_grad(set_to_none=True)
        train_loss, real_correct, fake_correct = 0.0, 0, 0
        for offset in range(0, len(labels), config['micro_batch_size']):
            batch_images = images[offset:offset + config['micro_batch_size']].to(device, non_blocking=True)
            batch_labels = labels[offset:offset + config['micro_batch_size']].to(device, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(batch_images).flatten()
                loss_sum = F.binary_cross_entropy_with_logits(logits, batch_labels, reduction='sum')
            (loss_sum / len(labels)).backward()
            train_loss += loss_sum.detach().item()
            predicted = logits.detach() >= 0
            real = batch_labels == 0
            fake = ~real
            real_correct += int(((~predicted) & real).sum())
            fake_correct += int((predicted & fake).sum())
        optimizer.step()
        scheduler.step()
        optimizer_steps += 1
        completed = block_end
        pair_count = block_end - block_start
        row = {
            'optimizer_step': optimizer_steps, 'completed_samples': completed,
            'block_start': block_start, 'block_end': block_end,
            'first_sample_index': block_indices[0],
            'last_sample_index': block_indices[-1],
            'train_loss': train_loss / (2 * pair_count),
            'train_real_accuracy': real_correct / pair_count,
            'train_fake_accuracy': fake_correct / pair_count,
            'learning_rate': learning_rate,
            'next_learning_rate': optimizer.param_groups[0]['lr'],
            'elapsed_seconds': time.monotonic() - started,
        }
        history.append(row)
        save_state(latest_path, model, optimizer, scheduler, completed, optimizer_steps,
                   config, data_identity, history)
        with metrics_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row) + '\n')
        if tracker:
            tracker_rng_state = rng_state()
            tracker.log({f'train/{key}': value for key, value in row.items()
                         if isinstance(value, (int, float)) and key != 'optimizer_step'},
                        step=optimizer_steps)
            restore_rng(tracker_rng_state)
        print(json.dumps(row), flush=True)

    if completed == config['num_images']:
        shutil.copyfile(latest_path, output / 'final.pt.tmp')
        os.replace(output / 'final.pt.tmp', output / 'final.pt')
        write_json({'complete': True, 'samples': completed, 'optimizer_steps': optimizer_steps,
                    'final_learning_rate': optimizer.param_groups[0]['lr'],
                    'checkpoint': str((output / 'final.pt').resolve())}, output / 'summary.json')
    else:
        print(f'Stopped at {completed}/{config["num_images"]} pairs due to --max-samples.', flush=True)
    if tracker:
        swanlab.finish()


if __name__ == '__main__':
    main()
