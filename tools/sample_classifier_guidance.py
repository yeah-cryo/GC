import argparse
import hashlib
import json
import os
import shutil
import textwrap
import time
from pathlib import Path

import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import save_image
from torchvision.transforms.functional import to_tensor

from aigi_detection.engine import seed_everything, write_json
from aigi_detection.datasets.coco_prompts import read_prompt_manifest
from aigi_detection.losses.losses import critic_logits
from aigi_detection.models.backbones import build_critic
from aigi_detection.models.modules.differentiable_sd import DifferentiableSD
from aigi_detection.models.modules.soft_prompt import encode_plain
from aigi_detection.models.modules.classifier_guidance import sample_guided


def resolve_seeds(config):
    if 'seeds' in config:
        seeds = list(config['seeds'])
    else:
        start = int(config.get('seed_start', 0))
        count = int(config['num_images'])
        seeds = list(range(start, start + count))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Seeds must be non-empty and unique.')
    return seeds


def read_metrics(path):
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
    keys = [(float(row['strength']), row['seed']) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f'Duplicate strength/seed rows in {path}')
    return rows


def write_json_atomic(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    write_json(value, temporary)
    os.replace(temporary, path)


def remake_comparison(output, config, seeds_per_page=4):
    metrics_path = output / 'metrics.jsonl'
    if not metrics_path.exists():
        raise FileNotFoundError(f'Missing completed metrics: {metrics_path}')
    rows = read_metrics(metrics_path)
    strengths = config['strengths']
    seeds = resolve_seeds(config)
    lookup = {(float(row['strength']), row['seed']): row for row in rows}
    expected = {(float(strength), seed) for strength in strengths for seed in seeds}
    if set(lookup) != expected:
        raise ValueError('Metrics must contain exactly one result for every configured strength and seed.')

    tile, label_height, margin, title_height = 256, 68, 12, 28
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
        title_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
    except OSError:
        font = title_font = ImageFont.load_default()
    page_paths = []
    for page_index, offset in enumerate(range(0, len(seeds), seeds_per_page), start=1):
        page_seeds = seeds[offset:offset + seeds_per_page]
        width = margin + len(strengths) * (tile + margin)
        height = title_height + margin + len(page_seeds) * (tile + label_height + margin)
        canvas = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((margin, 5),
                  f'Matched prompts and noise seeds — page {page_index}',
                  fill='black', font=title_font)
        for row_index, seed in enumerate(page_seeds):
            for column_index, strength in enumerate(strengths):
                left = margin + column_index * (tile + margin)
                top = title_height + margin + row_index * (tile + label_height + margin)
                image_path = output / f'strength_{strength:g}_seed_{seed}.png'
                with Image.open(image_path) as image:
                    canvas.paste(image.convert('RGB').resize((tile, tile), Image.Resampling.LANCZOS),
                                 (left, top))
                result = lookup[(float(strength), seed)]
                probability = result.get('saved_png_fake_probability', result['fake_probability'])
                caption = textwrap.shorten(result.get('prompt', config.get('prompt', '')),
                                           width=42, placeholder='…')
                draw.text((left, top + tile + 2), f'Seed {seed} | strength {strength:g}',
                          fill='black', font=font)
                draw.text((left, top + tile + 17), f'Fake probability: {probability:.2%}',
                          fill='black', font=font)
                draw.multiline_text((left, top + tile + 32), textwrap.fill(caption, width=38),
                                    fill='black', font=font, spacing=1)
        page_path = output / f'matched_comparison_page_{page_index:02d}.jpg'
        canvas.save(page_path, quality=95)
        page_paths.append(page_path.name)
        if page_index == 1:
            canvas.save(output / 'matched_comparison.jpg', quality=95)
    write_json({'layout': 'one matched seed per row; guidance strengths in columns',
                'strengths': strengths, 'seeds_per_page': seeds_per_page,
                'pages': page_paths,
                'score': 'saved_png_fake_probability when available, otherwise in-memory fake_probability'},
               output / 'matched_comparison_manifest.json')
    print(f'Remade matched_comparison.jpg and {len(page_paths)} matched pages under {output}', flush=True)


def fingerprint(models):
    result = hashlib.sha256()
    for i, model in enumerate(models):
        for name, tensor in model.state_dict().items():
            result.update(f'{i}:{name}'.encode())
            result.update(tensor.detach().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description='Low-noise predicted-x0 classifier guidance')
    parser.add_argument('--config', default='configs/experiments/sd14_classifier_guidance.yaml')
    parser.add_argument('--visualize-only', action='store_true',
                        help='Rebuild matched comparison pages from completed samples without generation.')
    parser.add_argument('--seeds-per-page', type=int, default=4)
    parser.add_argument('--resume', action='store_true',
                        help='Continue an interrupted run, skipping completed strength/seed pairs.')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    seeds = resolve_seeds(config)
    if not config['strengths'] or config['precision'] != 'bf16':
        raise ValueError('Specify at least one guidance strength, using BF16.')
    if 0 in config['strengths'] and config['strengths'][0] != 0:
        raise ValueError('When included, the unguided baseline must come first.')
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    if args.visualize_only:
        remake_comparison(output, config, args.seeds_per_page)
        return
    metrics_path = output / 'metrics.jsonl'
    existing_results = read_metrics(metrics_path)
    if existing_results and not args.resume:
        raise FileExistsError('This run has metrics. Pass --resume or use a new output directory.')
    configuration_path = output / 'configuration.yaml'
    if args.resume and configuration_path.exists():
        saved_config = yaml.safe_load(configuration_path.read_text())
        if saved_config != config:
            raise ValueError('Resume configuration differs from the saved run configuration.')
    elif not existing_results:
        configuration_path.write_text(yaml.safe_dump(config, sort_keys=False))
    if args.resume and (output / 'summary.json').exists():
        completed_summary = json.loads((output / 'summary.json').read_text())
        if completed_summary.get('complete'):
            print(f'Run already complete: {output}', flush=True)
            return
    seed_everything(42)
    torch.set_num_threads(2)
    if config.get('snapshot_critic', True):
        snapshot = output / 'critic.pt'
        if not snapshot.exists():
            shutil.copyfile(config['critic_checkpoint'], snapshot.with_suffix('.tmp'))
            os.replace(snapshot.with_suffix('.tmp'), snapshot)
    else:
        snapshot = Path(config['critic_checkpoint'])
    with snapshot.open('rb') as handle:
        critic_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
    identity = {'source': config['critic_checkpoint'], 'snapshot': str(snapshot),
                'sha256': critic_hash, 'type': config.get('critic_type', 'resnet50')}
    checkpoint = torch.load(snapshot, map_location='cpu', weights_only=False)
    if config.get('critic_type') == 'dinov3_original_mlp':
        from aigi_detection.models.backbones.dinov3_critic import DINOv3Critic
        implementation = output / 'dinov3_backbone_implementation.py'
        shutil.copyfile(config['dinov3_implementation'], implementation)
        critic = DINOv3Critic(implementation, config['dinov3_backbone'], checkpoint['original_head_state_dict']).cuda()
        with (Path(config['dinov3_backbone']) / 'model.safetensors').open('rb') as handle:
            identity['backbone_sha256'] = hashlib.file_digest(handle, 'sha256').hexdigest()
        identity.update(backbone=config['dinov3_backbone'], head='original_head_state_dict',
                        preprocessing='224x224 bilinear resize + ImageNet normalization',
                        precision='FP16 autocast, matching GGGT', label_mapping={'nature': 0, 'ai': 1})
        # Compare the adapter with GGGT's explicit normalized-CLS + two-layer head path.
        probe = torch.rand(1, 3, 224, 224, device='cuda', requires_grad=True)
        adapted = critic(probe)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            embedding = torch.nn.functional.normalize(critic.model(probe)['cls_token'].float(), p=2, dim=-1)
            direct = critic.fc2(torch.nn.functional.gelu(critic.fc1(embedding))).flatten().float()
        torch.testing.assert_close(adapted, direct, rtol=0, atol=0)
        gradient = torch.autograd.grad(adapted.sum(), probe)[0]
        assert torch.isfinite(gradient).all() and gradient.norm() > 0
        write_json({'native_head_matches_exactly': True, 'input_gradient_norm': gradient.norm().item(),
                    'parameters_frozen': all(not p.requires_grad and p.grad is None for p in critic.parameters())},
                   output / 'critic_audit.json')
        del probe, adapted, direct, embedding, gradient
    elif config.get('critic_type') == 'probe_dinov2_linear':
        from aigi_detection.models.backbones.probe_dinov2_critic import PROBEDINOv2Critic
        if set(checkpoint) != {'model_state_dict'}:
            raise ValueError(f'Unexpected PROBE DINOv2 checkpoint keys: {list(checkpoint)}')
        critic = PROBEDINOv2Critic(
            checkpoint['model_state_dict'], crop_size=config.get('critic_crop_size', 336)).cuda()
        identity.update(
            architecture='DINOv2-L/14 with 4 registers + linear 1024-to-1 head',
            preprocessing='ImageNet normalization; non-overlapping 336x336 crops; mean patch logit',
            precision='BF16 autocast for differentiable guidance and scoring',
            label_mapping={'nature': 0, 'ai': 1},
            source_repository='/mnt/e/repos/PROBE-AIGI-Detection')
        probe = torch.rand(1, 3, config['resolution'], config['resolution'],
                           device='cuda', requires_grad=True)
        adapted = critic.score_images(probe)
        gradient = torch.autograd.grad(adapted.sum(), probe)[0]
        if not torch.isfinite(gradient).all() or gradient.norm() <= 0:
            raise RuntimeError('PROBE DINOv2 critic did not provide a finite image gradient.')
        write_json({'input_gradient_norm': gradient.norm().item(),
                    'output_shape': list(adapted.shape),
                    'parameters_frozen': all(not p.requires_grad and p.grad is None
                                             for p in critic.parameters())},
                   output / 'critic_audit.json')
        del probe, adapted, gradient
    elif config.get('critic_type') == 'simlbr_dinov3_mlp':
        from aigi_detection.models.backbones.simlbr_critic import SimLBRCritic
        expected = {'state_dict', 'hyper_parameters'}
        if not expected.issubset(checkpoint):
            raise ValueError(f'Unexpected SimLBR checkpoint keys: {list(checkpoint)}')
        hyperparameters = checkpoint['hyper_parameters']
        if (hyperparameters.get('backbone') != 'dinov3'
                or hyperparameters.get('hidden_layers') != 2
                or hyperparameters.get('activation') != 'relu'):
            raise ValueError(f'Unsupported SimLBR architecture: {hyperparameters}')
        critic = SimLBRCritic(
            config['dinov3_backbone'], checkpoint['state_dict'],
            size=config.get('critic_image_size', 256)).cuda()
        identity.update(
            architecture='DINOv3-L/16 + ReLU MLP 1024-to-512-to-256-to-1',
            backbone=config['dinov3_backbone'], hyper_parameters=hyperparameters,
            preprocessing='256x256 bilinear resize; SimLBR RGB normalization',
            precision='BF16 autocast for differentiable guidance and scoring',
            label_mapping={'nature': 0, 'ai': 1},
            source_repository='/mnt/e/repos/SimLBR')
        probe = torch.rand(1, 3, config['resolution'], config['resolution'],
                           device='cuda', requires_grad=True)
        adapted = critic.score_images(probe)
        gradient = torch.autograd.grad(adapted.sum(), probe)[0]
        if not torch.isfinite(gradient).all() or gradient.norm() <= 0:
            raise RuntimeError('SimLBR critic did not provide a finite image gradient.')
        write_json({'input_gradient_norm': gradient.norm().item(),
                    'output_shape': list(adapted.shape),
                    'parameters_frozen': all(not p.requires_grad and p.grad is None
                                             for p in critic.parameters())},
                   output / 'critic_audit.json')
        del probe, adapted, gradient
    elif config.get('critic_type') == 'npr_resnet50':
        from aigi_detection.models.backbones.npr_critic import NPRCritic
        critic = NPRCritic(
            snapshot, resize_size=config.get('critic_resize_size', 256),
            crop_size=config.get('critic_crop_size', 224)).cuda()
        identity.update(
            architecture='NPR truncated ResNet-50 (layers 1-2) with binary linear head',
            preprocessing='256x256 bilinear resize; 224x224 center crop; ImageNet normalization',
            precision='BF16 SD generation; FP32 NPR critic',
            label_mapping={'nature': 0, 'ai': 1},
            source_repository='/mnt/e/repos/NPR-DeepfakeDetection')
        probe = torch.rand(1, 3, config['resolution'], config['resolution'],
                           device='cuda', requires_grad=True)
        adapted = critic.score_images(probe)
        gradient = torch.autograd.grad(adapted.sum(), probe)[0]
        if not torch.isfinite(gradient).all() or gradient.norm() <= 0:
            raise RuntimeError('NPR critic did not provide a finite image gradient.')
        write_json({'input_gradient_norm': gradient.norm().item(),
                    'output_shape': list(adapted.shape),
                    'parameters_frozen': all(not p.requires_grad and p.grad is None
                                             for p in critic.parameters())},
                   output / 'critic_audit.json')
        del probe, adapted, gradient
    elif config.get('critic_type') == 'effort_clip_l14':
        from aigi_detection.models.backbones.effort_critic import EffortCritic
        if not isinstance(checkpoint, dict) or 'module.head.weight' not in checkpoint:
            raise ValueError('Unexpected EFFORT checkpoint structure.')
        critic = EffortCritic(
            checkpoint, image_size=config.get('critic_image_size', 224)).cuda()
        identity.update(
            architecture='EFFORT CLIP ViT-L/14 with rank-one residual attention and 2-class head',
            preprocessing='224x224 bilinear resize; CLIP normalization',
            precision='BF16 autocast for differentiable guidance and scoring',
            label_mapping={'nature': 0, 'ai': 1},
            source_repository='/mnt/e/repos/Effort-AIGI-Detection')
        probe = torch.rand(1, 3, config['resolution'], config['resolution'],
                           device='cuda', requires_grad=True)
        adapted = critic.score_images(probe)
        gradient = torch.autograd.grad(adapted.sum(), probe)[0]
        if not torch.isfinite(gradient).all() or gradient.norm() <= 0:
            raise RuntimeError('EFFORT critic did not provide a finite image gradient.')
        write_json({'input_gradient_norm': gradient.norm().item(),
                    'output_shape': list(adapted.shape),
                    'parameters_frozen': all(not p.requires_grad and p.grad is None
                                             for p in critic.parameters()),
                    'effective_attention_weights_merged': True},
                   output / 'critic_audit.json')
        del probe, adapted, gradient
    elif config.get('critic_type') == 'safe_wavelet_resnet50':
        from aigi_detection.models.backbones.safe_critic import SAFECritic
        if not isinstance(checkpoint, dict) or set(checkpoint) != {'model'}:
            raise ValueError('Unexpected SAFE checkpoint structure.')
        critic = SAFECritic(
            checkpoint, crop_size=config.get('critic_crop_size', 256)).cuda()
        identity.update(
            architecture='SAFE bior1.3 diagonal-wavelet truncated ResNet-50 with 2-class head',
            preprocessing='256x256 center crop; one-level symmetric bior1.3 diagonal detail; no normalization',
            precision='BF16 SD generation; FP32 SAFE critic',
            label_mapping={'nature': 0, 'ai': 1},
            source_repository='/mnt/e/repos/SAFE')
        probe = torch.rand(1, 3, config['resolution'], config['resolution'],
                           device='cuda', requires_grad=True)
        adapted = critic.score_images(probe)
        gradient = torch.autograd.grad(adapted.sum(), probe)[0]
        if not torch.isfinite(gradient).all() or gradient.norm() <= 0:
            raise RuntimeError('SAFE critic did not provide a finite image gradient.')
        write_json({'input_gradient_norm': gradient.norm().item(),
                    'output_shape': list(adapted.shape),
                    'parameters_frozen': all(not p.requires_grad and p.grad is None
                                             for p in critic.parameters())},
                   output / 'critic_audit.json')
        del probe, adapted, gradient
    elif config.get('critic_type') == 'spai_spectral_vit':
        from aigi_detection.models.backbones.spai_critic import SPAICritic
        if not isinstance(checkpoint, dict) or 'model' not in checkpoint:
            raise ValueError('Unexpected SPAI checkpoint structure.')
        critic = SPAICritic(
            config['spai_repository'], checkpoint,
            config.get('spai_config')).cuda()
        identity.update(
            architecture='SPAI MFM ViT-B/16 spectral restoration detector with patch attention',
            preprocessing='native-resolution non-overlapping 224x224 patches; FFT low/high-frequency decomposition; ImageNet normalization inside SPAI',
            precision='BF16 autocast for differentiable guidance and scoring',
            label_mapping={'nature': 0, 'ai': 1},
            source_repository=config['spai_repository'],
            source_config=config.get('spai_config',
                                     str(Path(config['spai_repository']) / 'configs' / 'spai.yaml')))
        probe = torch.rand(1, 3, config['resolution'], config['resolution'],
                           device='cuda', requires_grad=True)
        adapted = critic.score_images(probe)
        gradient = torch.autograd.grad(adapted.sum(), probe)[0]
        if not torch.isfinite(gradient).all() or gradient.norm() <= 0:
            raise RuntimeError('SPAI critic did not provide a finite image gradient.')
        write_json({'input_gradient_norm': gradient.norm().item(),
                    'output_shape': list(adapted.shape),
                    'strict_checkpoint_load': True,
                    'checkpoint_parameter_count': sum(value.numel() for value in checkpoint['model'].values()),
                    'parameters_frozen': all(not p.requires_grad and p.grad is None
                                             for p in critic.parameters()),
                    'native_spai_no_grad_guard_disabled_for_input_gradient': True},
                   output / 'critic_audit.json')
        del probe, adapted, gradient
    else:
        if checkpoint['label_mapping'] != {'nature': 0, 'ai': 1}:
            raise ValueError('Expected fake=1, real=0 critic.')
        critic = build_critic().cuda().eval().requires_grad_(False)
        critic.load_state_dict(checkpoint['model'], strict=True)
    write_json(identity, output / 'critic_identity.json')
    del checkpoint
    print('Loading SD 1.4 for classifier-guided sampling...', flush=True)
    sd = DifferentiableSD(config['model_path'], steps=config['ddim_steps'], guidance=config['cfg_scale'], size=config['resolution'])
    if config.get('prompt_manifest'):
        prompt_records = read_prompt_manifest(config['prompt_manifest'], len(seeds))
        if len(prompt_records) != len(seeds):
            raise ValueError('The prompt manifest must contain at least one prompt per seed.')
    else:
        prompt_records = [{'prompt_index': index, 'caption': config['prompt']}
                          for index in range(len(seeds))]
    prompt_snapshot = [{'seed': seed, **record} for seed, record in zip(seeds, prompt_records)]
    prompt_snapshot_path = output / 'prompts.jsonl'
    if args.resume and prompt_snapshot_path.exists():
        saved_prompts = read_prompt_manifest(prompt_snapshot_path)
        if saved_prompts != prompt_snapshot:
            raise ValueError('Resume prompts differ from the saved prompt snapshot.')
    else:
        with prompt_snapshot_path.open('w', encoding='utf-8') as handle:
            for record in prompt_snapshot:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    with torch.no_grad():
        empty = encode_plain(sd.text_encoder, sd.tokenizer, '')
    models = (*sd.models(), critic)
    before = fingerprint(models)
    state_path = output / 'run_state.json'
    if args.resume and state_path.exists():
        saved_state = json.loads(state_path.read_text())
        if saved_state['model_fingerprint'] != before:
            raise ValueError('Frozen model fingerprint differs from the interrupted run.')
    write_json_atomic({'complete': False, 'model_fingerprint': before,
                       'configured_images': len(seeds) * len(config['strengths']),
                       'completed_images': len(existing_results)}, state_path)

    results = list(existing_results)
    completed = {(float(row['strength']), row['seed']): row for row in results}
    make_full_grids = len(seeds) <= 100
    condition_cache = {} if make_full_grids else None
    all_images = []
    for strength in config['strengths']:
        row_images, preview_images = [], []
        for index, seed in enumerate(seeds):
            key = (float(strength), seed)
            image_path = output / f'strength_{strength:g}_seed_{seed}.png'
            if key in completed:
                if not image_path.exists():
                    raise FileNotFoundError(f'Metrics exist but image is missing: {image_path}')
                if make_full_grids or index < 16:
                    with Image.open(image_path) as saved:
                        saved_tensor = to_tensor(saved.convert('RGB')).unsqueeze(0)
                    if make_full_grids:
                        row_images.append(saved_tensor)
                    if index < 16:
                        preview_images.append(saved_tensor)
                continue
            if condition_cache is not None and index in condition_cache:
                condition = condition_cache[index]
            else:
                with torch.no_grad():
                    condition = encode_plain(sd.text_encoder, sd.tokenizer, prompt_records[index]['caption'])
                if condition_cache is not None:
                    condition_cache[index] = condition
            started = time.monotonic()
            image, steps, estimates = sample_guided(
                sd, critic, condition, empty, seed, strength,
                config['max_guidance_timestep'], config['gradient_rms_clip'],
                capture_estimates=(index == 0 and strength > 0))
            with torch.no_grad():
                logit = critic_logits(critic, image).item()
                if strength == 0:
                    baseline_rmse = 0.0
                elif 0 in config['strengths']:
                    baseline_path = output / f'strength_0_seed_{seed}.png'
                    with Image.open(baseline_path) as baseline_saved:
                        baseline = to_tensor(baseline_saved.convert('RGB')).unsqueeze(0)
                    baseline_rmse = (image.cpu() - baseline).square().mean().sqrt().item()
                else:
                    baseline_rmse = None
                row = {'strength': strength, 'seed': seed,
                       'prompt_index': prompt_records[index]['prompt_index'],
                       'annotation_id': prompt_records[index].get('annotation_id'),
                       'image_id': prompt_records[index].get('image_id'),
                       'prompt': prompt_records[index]['caption'], 'fake_logit': logit,
                       'fake_probability': torch.tensor(logit).sigmoid().item(), 'classified_real': logit < 0,
                       'saturated_pixel_fraction': ((image < 0.01) | (image > 0.99)).float().mean().item(),
                       'guided_steps': sum(step['guided'] for step in steps), 'seconds': time.monotonic() - started,
                       'pixel_rmse_from_baseline': baseline_rmse}
            if strength == 0 and index == 0:
                with torch.no_grad():
                    reference = sd.generate(condition, empty, seed)
                torch.testing.assert_close(image, reference, rtol=0, atol=0)
            save_image(image, image_path)
            with Image.open(image_path) as saved, torch.no_grad():
                if hasattr(critic, 'score_pil'):
                    saved_logit = critic.score_pil(saved.convert('RGB')).item()
                else:
                    saved_logit = critic_logits(
                        critic, to_tensor(saved.convert('RGB')).unsqueeze(0).to(sd.device)).item()
            row.update(saved_png_fake_probability=torch.tensor(saved_logit).sigmoid().item(),
                       saved_png_classified_real=saved_logit < 0)
            write_json(steps, output / f'strength_{strength:g}_seed_{seed}_steps.json')
            if estimates:
                save_image(torch.cat(estimates), output / f'strength_{strength:g}_seed_{seed}_x0.jpg', nrow=4)
            cpu_image = image.detach().cpu()
            if make_full_grids:
                row_images.append(cpu_image)
            if index < 16:
                preview_images.append(cpu_image)
            results.append(row)
            completed[key] = row
            with metrics_path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            if len(results) % 25 == 0:
                write_json_atomic({'complete': False, 'model_fingerprint': before,
                                   'configured_images': len(seeds) * len(config['strengths']),
                                   'completed_images': len(results), 'last_strength': strength,
                                   'last_seed': seed}, state_path)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        if make_full_grids:
            save_image(torch.cat(row_images), output / f'strength_{strength:g}.png',
                       nrow=min(10, len(seeds)))
            if len(row_images) > 16:
                for offset in range(0, len(row_images), 16):
                    save_image(torch.cat(row_images[offset:offset + 16]),
                               output / f'strength_{strength:g}_page_{offset // 16 + 1:02d}.jpg', nrow=4)
            all_images.extend([
                torch.nn.functional.interpolate(image, (128, 128), mode='bilinear', align_corners=False)
                for image in row_images
            ] if len(seeds) > 16 else row_images)
        elif preview_images:
            save_image(torch.cat(preview_images), output / f'strength_{strength:g}_preview.jpg', nrow=4)
    if make_full_grids:
        save_image(torch.cat(all_images), output / 'comparison.jpg', nrow=min(10, len(seeds)))
    assert all(parameter.grad is None and not parameter.requires_grad
               for model in models for parameter in model.parameters())
    after = fingerprint(models)
    assert before == after, 'Frozen model weights or buffers changed'
    summary = []
    for strength in config['strengths']:
        strength_rows = [row for row in results if float(row['strength']) == float(strength)]
        deviations = [row['pixel_rmse_from_baseline'] for row in strength_rows
                      if row['pixel_rmse_from_baseline'] is not None]
        summary.append({'strength': strength, 'images': len(strength_rows),
                        'classified_real_count': sum(row['classified_real'] for row in strength_rows),
                        'mean_fake_probability': sum(row['fake_probability'] for row in strength_rows) / len(strength_rows),
                        'fooling_rate': sum(row['classified_real'] for row in strength_rows) / len(strength_rows),
                        'saved_png_classified_real_count': sum(row['saved_png_classified_real'] for row in strength_rows),
                        'saved_png_fooling_rate': sum(row['saved_png_classified_real'] for row in strength_rows) / len(strength_rows),
                        'mean_saved_png_fake_probability': sum(row['saved_png_fake_probability'] for row in strength_rows) / len(strength_rows),
                        'mean_pixel_rmse_from_baseline': sum(deviations) / len(deviations) if deviations else None})
    summary_payload = {'complete': True, 'summary': summary, 'rows': config['strengths'],
                       'columns': seeds if len(seeds) <= 1000 else None,
                       'seed_start': seeds[0], 'seed_count': len(seeds),
                       'model_fingerprint': before, 'frozen_models_unchanged': True,
                       'zero_guidance_matches_reference': True if 0 in config['strengths'] else None,
                       'quality_review_required': True}
    write_json_atomic(summary_payload, output / 'summary.json')
    write_json_atomic({'complete': True, 'model_fingerprint': before,
                       'configured_images': len(seeds) * len(config['strengths']),
                       'completed_images': len(results)}, state_path)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
