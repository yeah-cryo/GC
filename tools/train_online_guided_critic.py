import argparse
import hashlib
import json
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
from torchvision.utils import save_image

from aigi_detection.datasets import TrainTransform
from aigi_detection.datasets.coco_prompts import read_prompt_manifest
from aigi_detection.engine import atomic_save, restore_rng, rng_state, seed_everything, write_json
from aigi_detection.losses.losses import critic_logits
from aigi_detection.models.backbones import build_critic
from aigi_detection.models.modules.classifier_guidance import sample_guided
from aigi_detection.models.modules.differentiable_sd import DifferentiableSD
from aigi_detection.models.modules.soft_prompt import encode_plain
from aigi_detection.training import build_replay


LABELS = {'nature': 0, 'ai': 1}


def file_sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def guidance_strength(index, config):
    warmup = int(config['warmup_images'])
    ramp = int(config['guidance_ramp_images'])
    target = float(config['guidance_strength'])
    if index < warmup:
        return 0.0
    if ramp and index < warmup + ramp:
        return target * (index - warmup + 1) / ramp
    return target


def real_path(root, image_id):
    return Path(root) / f'COCO_train2014_{int(image_id):012d}.jpg'


def load_augmented(path, transform, label):
    with Image.open(path) as image:
        return transform(image.convert('RGB'), label)


def write_generation_rows(path, rows):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    os.replace(temporary, path)


def save_state(path, model, optimizer, completed, steps, config, manifest_hash, history, replay):
    state = {
        'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'completed_samples': completed, 'optimizer_steps': steps,
        'configuration': config, 'label_mapping': LABELS,
        'prompt_manifest_sha256': manifest_hash, 'rng_state': rng_state(),
        'replay': replay.state_dict() if replay else None,
        'metrics': history[-1] if history else None,
        'history': history, 'architecture': 'resnet50',
        'output_semantics': 'one logit; sigmoid = P(fake)',
    }
    atomic_save(state, path)


def main():
    parser = argparse.ArgumentParser(
        description='Train one ResNet critic with online SD1.4 classifier-guided hard negatives.')
    parser.add_argument('--config', default='configs/experiments/resnet50_online_guided_coco20k.yaml')
    parser.add_argument('--resume', nargs='?', const='auto')
    parser.add_argument('--max-samples', type=int,
                        help='Development aid: stop after this many total samples.')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('A BF16 CUDA GPU is required.')
    if config['precision'] != 'bf16':
        raise ValueError('This experiment requires BF16.')
    if config['pair_batch_size'] < 1 or config['micro_batch_size'] < 1:
        raise ValueError('Batch sizes must be positive.')
    if config['guidance_strength'] < 0 or config['warmup_images'] < 0 or config['guidance_ramp_images'] < 0:
        raise ValueError('Guidance schedule values must be nonnegative.')

    seed_everything(config['seed'])
    torch.set_num_threads(2)
    device = torch.device('cuda')
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    fake_dir = output / 'generated'
    fake_dir.mkdir(exist_ok=True)
    latest_path = output / 'latest.pt'
    if args.resume == 'auto':
        args.resume = str(latest_path)
    if not args.resume and latest_path.exists():
        raise FileExistsError(f'Existing run found at {output}; pass --resume.')

    prompt_source = Path(config['prompt_manifest'])
    prompt_records = read_prompt_manifest(prompt_source, config['num_images'])
    if len(prompt_records) != config['num_images']:
        raise ValueError('Prompt manifest is shorter than num_images.')
    image_ids = [int(row['image_id']) for row in prompt_records]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError('Online experiment requires one record per unique COCO image ID.')
    if [row['prompt_index'] for row in prompt_records] != list(range(len(prompt_records))):
        raise ValueError('Prompt indices must be consecutive from zero.')
    if not Path(config['coco_images_root']).is_dir():
        raise FileNotFoundError(f"COCO image root is missing: {config['coco_images_root']}")
    manifest_hash = file_sha256(prompt_source)
    manifest_snapshot = output / 'prompts.jsonl'
    if manifest_snapshot.exists():
        if file_sha256(manifest_snapshot) != manifest_hash:
            raise ValueError('Saved prompt manifest differs from configured manifest.')
    else:
        shutil.copyfile(prompt_source, manifest_snapshot)

    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    if checkpoint:
        if checkpoint['configuration'] != config or checkpoint['prompt_manifest_sha256'] != manifest_hash:
            raise ValueError('Resume configuration or prompt manifest differs.')
        if checkpoint['label_mapping'] != LABELS:
            raise ValueError('Resume label mapping differs.')
    replay = build_replay(config)
    if checkpoint and replay:
        if checkpoint.get('replay') is None:
            raise ValueError('Replay is enabled but the checkpoint has no replay state.')
        replay.load_state_dict(checkpoint['replay'])
        if replay.seen != int(checkpoint['completed_samples']):
            raise ValueError('Replay state does not match completed samples.')
    model = build_critic(None if checkpoint else config['pretrained']).to(device)
    if checkpoint:
        model.load_state_dict(checkpoint['model'], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                  weight_decay=config['weight_decay'], betas=tuple(config['betas']),
                                  eps=config['epsilon'])
    completed, optimizer_steps, history = 0, 0, []
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        completed = int(checkpoint['completed_samples'])
        optimizer_steps = int(checkpoint['optimizer_steps'])
        history = checkpoint.get('history', [])
        restore_rng(checkpoint['rng_state'])
    del checkpoint

    generation_path = output / 'generation_metrics.jsonl'
    generation_rows = []
    if generation_path.exists():
        saved_generation_rows = [json.loads(line) for line in generation_path.read_text(encoding='utf-8').splitlines()
                                 if line.strip()]
        generation_rows = [row for row in saved_generation_rows if row['prompt_index'] < completed]
        write_generation_rows(generation_path, generation_rows)
    metrics_path = output / 'metrics.jsonl'
    metrics_path.write_text(''.join(json.dumps(row) + '\n' for row in history), encoding='utf-8')
    (output / 'configuration.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    write_json({'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                'packages': {name: version(name) for name in ('torchvision', 'diffusers', 'transformers',
                                                               'numpy', 'Pillow', 'PyYAML')},
                'gpu': torch.cuda.get_device_name(), 'label_mapping': LABELS},
               output / 'environment.json')
    write_json({'source': str(prompt_source.resolve()), 'copied_to': str(manifest_snapshot.resolve()),
                'sha256': manifest_hash, 'records': len(prompt_records),
                'unique_image_ids': len(set(image_ids)),
                'selection': 'first caption annotation per unique COCO image ID'},
               output / 'manifest_identity.json')
    if not latest_path.exists():
        save_state(latest_path, model, optimizer, completed, optimizer_steps,
                   config, manifest_hash, history, replay)

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

    print('Loading frozen Stable Diffusion 1.4...', flush=True)
    sd = DifferentiableSD(config['model_path'], steps=config['ddim_steps'],
                          guidance=config['cfg_scale'], size=config['resolution'])
    with torch.no_grad():
        unconditional = encode_plain(sd.text_encoder, sd.tokenizer, '')
    transform = TrainTransform(config['crop_size'])
    total = min(config['num_images'], args.max_samples or config['num_images'])
    started = time.monotonic()

    while completed < total:
        block_start = completed
        block_end = min(total, block_start + config['pair_batch_size'])
        model.eval().requires_grad_(False)
        block_generation = []
        print(f'Generating online hard-negative block {block_start}:{block_end}...', flush=True)
        for index in range(block_start, block_end):
            record = prompt_records[index]
            strength = guidance_strength(index, config)
            path = fake_dir / f'fake_{index:05d}.png'
            generated_at = time.monotonic()
            with torch.no_grad():
                condition = encode_plain(sd.text_encoder, sd.tokenizer, record['caption'])
            image, step_logs, _ = sample_guided(
                sd, model, condition, unconditional, config['seed_start'] + index, strength,
                config['max_guidance_timestep'], config['gradient_rms_clip'])
            with torch.no_grad():
                preupdate_logit = critic_logits(model, image).item()
            save_image(image, path)
            guided = [row for row in step_logs if row['guided']]
            row = {
                'prompt_index': index, 'annotation_id': record['annotation_id'],
                'image_id': record['image_id'], 'caption': record['caption'],
                'seed': config['seed_start'] + index, 'strength': strength,
                'fake_path': str(path), 'preupdate_fake_logit': preupdate_logit,
                'preupdate_fake_probability': torch.tensor(preupdate_logit).sigmoid().item(),
                'guided_steps': len(guided),
                'mean_guidance_gradient_rms': (sum(x['gradient_rms'] for x in guided) / len(guided)
                                               if guided else 0.0),
                'generation_seconds': time.monotonic() - generated_at,
            }
            generation_rows.append(row)
            block_generation.append(row)
            print(json.dumps({key: row[key] for key in ('prompt_index', 'strength',
                                                        'preupdate_fake_probability', 'generation_seconds')}),
                  flush=True)

        # The SD graph is gone. Train the same detector on current and replayed pairs.
        model.requires_grad_(True).train()
        current_real, current_fake = [], []
        for index in range(block_start, block_end):
            record = prompt_records[index]
            current_real.append(load_augmented(real_path(config['coco_images_root'], record['image_id']),
                                               transform, 0))
            current_fake.append(load_augmented(fake_dir / f'fake_{index:05d}.png', transform, 1))
        replay_indices = replay.sample() if replay else []
        replay_real, replay_fake = [], []
        for index in replay_indices:
            record = prompt_records[index]
            replay_real.append(load_augmented(real_path(config['coco_images_root'], record['image_id']),
                                              transform, 0))
            replay_fake.append(load_augmented(fake_dir / f'fake_{index:05d}.png', transform, 1))

        current_pairs, replay_pairs = len(current_real), len(replay_real)
        replay_weight = replay.weight if replay_pairs else 0.0
        images = torch.stack(current_real + current_fake + replay_real + replay_fake)
        labels = torch.cat((torch.zeros(current_pairs), torch.ones(current_pairs),
                            torch.zeros(replay_pairs), torch.ones(replay_pairs)))
        groups = torch.cat((torch.zeros(2 * current_pairs, dtype=torch.long),
                            torch.ones(2 * replay_pairs, dtype=torch.long)))
        weights = torch.cat((
            torch.full((2 * current_pairs,), (1.0 - replay_weight) / (2 * current_pairs)),
            torch.full((2 * replay_pairs,), replay_weight / (2 * replay_pairs))
            if replay_pairs else torch.empty(0),
        ))
        order = torch.randperm(len(labels))
        images, labels, groups, weights = (value[order] for value in (images, labels, groups, weights))
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        group_loss = [0.0, 0.0]
        group_real_correct = [0, 0]
        group_fake_correct = [0, 0]
        for offset in range(0, len(labels), config['micro_batch_size']):
            batch_images = images[offset:offset + config['micro_batch_size']].to(device, non_blocking=True)
            batch_labels = labels[offset:offset + config['micro_batch_size']].to(device, non_blocking=True)
            batch_groups = groups[offset:offset + config['micro_batch_size']].to(device, non_blocking=True)
            batch_weights = weights[offset:offset + config['micro_batch_size']].to(device, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(batch_images).flatten()
                losses = F.binary_cross_entropy_with_logits(logits, batch_labels, reduction='none')
                weighted_loss = (losses * batch_weights).sum()
            weighted_loss.backward()
            train_loss += weighted_loss.detach().item()
            predicted = logits.detach() >= 0
            real = batch_labels == 0
            fake = ~real
            for group in (0, 1):
                selected = batch_groups == group
                group_loss[group] += losses.detach()[selected].sum().item()
                group_real_correct[group] += int(((~predicted) & real & selected).sum())
                group_fake_correct[group] += int((predicted & fake & selected).sum())
        if config.get('gradient_clip_norm'):
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm']).item()
        else:
            gradient_norm = None
        optimizer.step()
        optimizer_steps += 1
        completed = block_end
        pair_count = block_end - block_start
        if replay:
            replay.add(range(block_start, block_end))
        row = {
            'optimizer_step': optimizer_steps, 'completed_samples': completed,
            'block_start': block_start, 'block_end': block_end,
            'train_loss': train_loss,
            'train_current_loss': group_loss[0] / (2 * current_pairs),
            'train_real_accuracy': group_real_correct[0] / current_pairs,
            'train_fake_accuracy': group_fake_correct[0] / current_pairs,
            'replay_pairs': replay_pairs,
            'replay_weight': replay_weight,
            'replay_loss': group_loss[1] / (2 * replay_pairs) if replay_pairs else None,
            'replay_real_accuracy': group_real_correct[1] / replay_pairs if replay_pairs else None,
            'replay_fake_accuracy': group_fake_correct[1] / replay_pairs if replay_pairs else None,
            'replay_buffer_size': len(replay.indices) if replay else 0,
            'replay_seen': replay.seen if replay else 0,
            'mean_preupdate_fake_probability': sum(x['preupdate_fake_probability'] for x in block_generation) / pair_count,
            'mean_strength': sum(x['strength'] for x in block_generation) / pair_count,
            'gradient_norm': gradient_norm, 'elapsed_seconds': time.monotonic() - started,
        }
        history.append(row)
        write_generation_rows(generation_path, generation_rows)
        save_state(latest_path, model, optimizer, completed, optimizer_steps,
                   config, manifest_hash, history, replay)
        with metrics_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row) + '\n')
        if tracker:
            tracker.log({f'train/{key}': value for key, value in row.items()
                         if isinstance(value, (int, float)) and key not in ('optimizer_step',)},
                        step=optimizer_steps)
        print(json.dumps(row), flush=True)

    if completed == config['num_images']:
        shutil.copyfile(latest_path, output / 'final.pt.tmp')
        os.replace(output / 'final.pt.tmp', output / 'final.pt')
        write_json({'complete': True, 'samples': completed, 'optimizer_steps': optimizer_steps,
                    'checkpoint': str((output / 'final.pt').resolve())}, output / 'summary.json')
    else:
        print(f'Stopped at {completed}/{config["num_images"]} samples due to --max-samples.', flush=True)
    if tracker:
        swanlab.finish()


if __name__ == '__main__':
    main()
