"""Caption-free SD1.4 soft-prompt pilot. Only the prompt tensor is optimized."""
import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import torch
import yaml
from torch.nn import functional as F
from torchvision.utils import save_image

from aigi_detection.engine import atomic_save, seed_everything, write_json
from aigi_detection.losses.losses import PerceptualFeatures, critic_logits, perceptual_loss
from aigi_detection.models.backbones import build_critic
from aigi_detection.models.modules.differentiable_sd import DifferentiableSD
from aigi_detection.models.modules.soft_prompt import SoftPrompt, encode_plain, prompt_update_stats


def digest_file(path):
    with open(path, 'rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def fingerprint(models):
    digest = hashlib.sha256()
    for i, model in enumerate(models):
        for name, tensor in model.state_dict().items():
            digest.update(f'{i}:{name}'.encode())
            digest.update(tensor.detach().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def new_prompt(sd, config):
    return SoftPrompt(sd.text_encoder, sd.tokenizer, config['prefix'], config['soft_tokens'],
                      config['initialization_seed'], config['initialization_noise_std']).to(sd.device)


def objective(sd, prompt, critic, empty, seed, trace=None, perceptual=None, prefix=None,
              perceptual_weight=0.0, components=None):
    if perceptual is not None:
        with torch.no_grad():
            reference_features = perceptual(sd.generate(prefix, empty, seed))
    image = sd.generate(prompt(sd.text_encoder), empty, seed, trace=trace)
    logits = critic_logits(critic, image)
    bce = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
    perc = perceptual_loss(perceptual(image), reference_features) if perceptual is not None else bce.new_zeros(())
    loss = bce + perceptual_weight * perc
    if components is not None:
        components.update(bce=bce.detach().item(), perceptual=perc.detach().item())
    return loss, logits


def audit(sd, prompt, critic, perceptual, prefix, empty, config, output):
    print('Auditing full 35-step gradient path and frozen weights...', flush=True)
    models = (*sd.models(), critic, perceptual)
    before = fingerprint(models)
    original = prompt.soft_tokens.detach().clone()
    assert sum(p.numel() for p in prompt.parameters()) == 12288
    assert all(not p.requires_grad for model in models for p in model.parameters())
    tokenized = sd.tokenizer(config['prefix'], padding='max_length', max_length=77, truncation=True, return_tensors='pt')
    with torch.no_grad():
        kwargs = {'input_ids': tokenized['input_ids'].to(sd.device)}
        if getattr(sd.text_encoder.config, 'use_attention_mask', False):
            kwargs['attention_mask'] = tokenized['attention_mask'].to(sd.device)
        ordinary = sd.text_encoder(**kwargs).last_hidden_state
        torch.testing.assert_close(prefix, ordinary, rtol=0, atol=0)
        conditioning = prompt(sd.text_encoder)
        atomic_save(prompt.state_dict(), output / 'audit_prompt.pt')
        restored = new_prompt(sd, config)
        restored.load_state_dict(torch.load(output / 'audit_prompt.pt', weights_only=True, map_location=sd.device))
        restored_conditioning = restored(sd.text_encoder)
        torch.testing.assert_close(conditioning, restored_conditioning, rtol=0, atol=0)
        image1 = sd.generate(conditioning, empty, 31001)
        image2 = sd.generate(restored_conditioning, empty, 31001)
        torch.testing.assert_close(image1, image2, rtol=0, atol=1e-5)
        save_image(image1, output / 'audit_initialized.png')
        reproduction_error = (image1 - image2).abs().max().item()
    trace = {}
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    loss, logits = objective(sd, prompt, critic, empty, 31002, trace, perceptual, prefix, config['lambda_perceptual'])
    loss.backward()
    gradient = prompt.soft_tokens.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.norm() > 0
    assert all(p.grad is None for model in models for p in model.parameters())
    assert all(value > 0 for value in trace.values()) and len(trace) == 3
    optimizer = torch.optim.AdamW([prompt.soft_tokens], lr=config['learning_rate'], weight_decay=0)
    torch.nn.utils.clip_grad_norm_([prompt.soft_tokens], config['gradient_clip'])
    optimizer.step()
    assert not torch.equal(original, prompt.soft_tokens)
    after = fingerprint(models)
    assert before == after, 'A frozen parameter or model buffer changed'
    with torch.no_grad():
        torch.testing.assert_close(encode_plain(sd.text_encoder, sd.tokenizer, config['prefix']), prefix, rtol=0, atol=0)
        prompt.soft_tokens.copy_(original)
    prompt.soft_tokens.grad = None
    report = {'passed': True, 'trainable_parameters': 12288, 'model_fingerprint': before,
              'frozen_models_unchanged': True, 'prefix_conditioning_matches': True,
              'reload_conditioning_matches': True, 'reload_image_max_error': reproduction_error,
              'gradient_mode': 'full_35_step_checkpointed', 'gradient_norm': gradient.norm().item(), **trace,
              'seconds': time.monotonic() - started, 'peak_gpu_allocated_gib': torch.cuda.max_memory_allocated() / 2**30}
    write_json(report, output / 'audit.json')
    print(json.dumps(report), flush=True)


@torch.no_grad()
def evaluate(sd, prompt, critic, perceptual, prefix, empty, seeds, output, step, group):
    """Rows are prefix / initialized / trained; columns use identical seeds."""
    conditions = {'prefix': prefix, 'initialized': prompt(sd.text_encoder, initialized=True),
                  'trained': prompt(sd.text_encoder)}
    images, features, metrics = {}, {}, {}
    for name, conditioning in conditions.items():
        images[name], features[name], logits, deviations, saturations = [], [], [], [], []
        for index, seed in enumerate(seeds):
            image = sd.generate(conditioning, empty, seed)
            phi = perceptual(image)
            features[name].append(tuple(f.cpu() for f in phi))
            images[name].append(image.cpu())
            logits.append(critic_logits(critic, image).item())
            saturations.append(((image < 0.01) | (image > 0.99)).float().mean().item())
            if name != 'prefix':
                deviations.append(perceptual_loss(features[name][-1], features['prefix'][index]).item())
        diversity = [perceptual_loss(features[name][i], features[name][j]).item()
                     for i in range(len(seeds)) for j in range(i)]
        scores = torch.tensor(logits)
        metrics[name] = {'mean_fake_logit': scores.mean().item(), 'mean_fake_probability': scores.sigmoid().mean().item(),
                         'critic_fooling_rate': (scores < 0).float().mean().item(), 'logits': logits,
                         'perceptual_deviation': sum(deviations) / max(1, len(deviations)),
                         'perceptual_diversity': sum(diversity) / max(1, len(diversity)),
                         'saturated_pixel_fraction': sum(saturations) / len(saturations)}
    grid = torch.cat([image for name in conditions for image in images[name]])
    save_image(grid, output / f'{group}_{step:04d}.jpg', nrow=len(seeds))
    report = {'step': step, 'seed_group': group, 'seeds': seeds,
              'grid_rows': list(conditions), 'conditions': metrics,
              'quality_note': 'Perceptual/saturation/diversity proxies; visual review is required.'}
    write_json(report, output / f'{group}_{step:04d}.json')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments/sd14_soft_prompt.yaml')
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--resume')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if config.get('objective') != 'bce_perceptual' or config['lambda_perceptual'] <= 0:
        raise ValueError('Use the BCE-plus-perceptual configuration.')
    if config['gradient_mode'] != 'full_checkpointed' or config['batch_size'] != 1:
        raise ValueError('Only full-gradient, batch-one pilot is implemented.')
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'latest.pt').exists() and not args.resume and not args.audit_only:
        raise FileExistsError('Use --resume to continue an existing pilot.')
    seed_everything(config['initialization_seed'])
    torch.set_num_threads(2)
    train_seeds = list(range(config['train_seed_start'], config['train_seed_start'] + config['updates']))
    assert not set(train_seeds) & (set(config['diagnostic_seeds']) | set(config['heldout_seeds']))
    assert not set(config['diagnostic_seeds']) & set(config['heldout_seeds'])
    write_json({'training': train_seeds, 'diagnostics': config['diagnostic_seeds'], 'heldout': config['heldout_seeds'],
                'audit': [31001, 31002]}, output / 'seeds.json')
    snapshot = output / 'critic.pt'
    if not snapshot.exists():
        temporary = snapshot.with_suffix('.tmp')
        shutil.copyfile(config['critic_checkpoint'], temporary)
        os.replace(temporary, snapshot)
    identity = {'source': config['critic_checkpoint'], 'snapshot': str(snapshot), 'sha256': digest_file(snapshot),
                'architecture': config['critic_architecture']}
    write_json(identity, output / 'critic_identity.json')
    (output / 'configuration.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    state = torch.load(snapshot, map_location='cpu', weights_only=False)
    if state['label_mapping'] != {'nature': 0, 'ai': 1} or config['critic_architecture'] != 'resnet50_full_finetune':
        raise ValueError('Expected the selected fully fine-tuned binary ResNet-50 critic.')
    critic = build_critic().cuda().eval().requires_grad_(False)
    critic.load_state_dict(state['model'], strict=True)
    del state
    perceptual = PerceptualFeatures(config['perceptual_weights']).cuda()
    print('Loading local SD 1.4 components...', flush=True)
    sd = DifferentiableSD(config['model_path'], steps=config['ddim_steps'], guidance=config['guidance_scale'], size=config['resolution'])
    prompt = new_prompt(sd, config)
    with torch.no_grad():
        prefix = encode_plain(sd.text_encoder, sd.tokenizer, config['prefix'])
        empty = encode_plain(sd.text_encoder, sd.tokenizer, '')
    if not args.resume:
        audit(sd, prompt, critic, perceptual, prefix, empty, config, output)
    if args.audit_only:
        return
    optimizer = torch.optim.AdamW([prompt.soft_tokens], lr=config['learning_rate'], weight_decay=config['weight_decay'])
    start, best, stagnant = 1, float('inf'), 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        if checkpoint['configuration'] != config or checkpoint['critic_sha256'] != identity['sha256']:
            raise ValueError('Resume configuration or critic identity differs.')
        prompt.load_state_dict(checkpoint['prompt'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start, best = checkpoint['step'] + 1, checkpoint['best_candidate_logit']
        stagnant = checkpoint.get('early_stopping_counter', 0)
        if checkpoint.get('stop_reason') == 'prompt_converged':
            print('Prompt has already reached the convergence stopping criterion.', flush=True)
            return
    tracker = None
    if os.environ.get('SWANLAB_API_KEY'):
        import swanlab
        tracker = swanlab.init(project='aigi-detection', experiment_name='sd14-soft-prompt-bce-perceptual', config=config,
                               public=False, mode='cloud', logdir=str(output / 'swanlab'))
        write_json({'url': tracker.url, 'id': tracker.id}, output / 'swanlab_run.json')
    if start == 1:
        for group in ('diagnostic', 'heldout'):
            evaluate(sd, prompt, critic, perceptual, prefix, empty, config[f'{group}_seeds'], output, 0, group)
    for step in range(start, config['updates'] + 1):
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        seed = train_seeds[step - 1]
        components = {}
        loss, logits = objective(sd, prompt, critic, empty, seed, perceptual=perceptual, prefix=prefix,
                                 perceptual_weight=config['lambda_perceptual'], components=components)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([prompt.soft_tokens], config['gradient_clip'], error_if_nonfinite=True)
        before_update = prompt.soft_tokens.detach().clone()
        optimizer.step()
        stopping = config['early_stopping']
        update_stats = prompt_update_stats(before_update, prompt.soft_tokens, stopping['absolute_tolerance'], stopping['relative_tolerance'])
        stagnant = stagnant + 1 if step >= stopping['minimum_updates'] and update_stats['near_stationary'] else 0
        converged = stagnant >= stopping['patience']
        row = {'step': step, 'seed': seed, 'loss': loss.item(), **components,
               'weighted_perceptual': config['lambda_perceptual'] * components['perceptual'],
               'lambda_perceptual': config['lambda_perceptual'],
               'fake_logit': logits.item(), 'fake_probability': logits.sigmoid().item(),
               'gradient_norm_before_clip': norm.item(),
               'seconds': time.monotonic() - started, **update_stats, 'early_stopping_counter': stagnant}
        with (output / 'losses.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
        if tracker:
            tracker.log({key: value for key, value in row.items() if isinstance(value, (int, float))}, step=step)
        selected = False
        if step % config['save_every'] == 0 or step == config['updates'] or converged:
            evaluate(sd, prompt, critic, perceptual, prefix, empty, config['diagnostic_seeds'], output, step, 'diagnostic')
            report = evaluate(sd, prompt, critic, perceptual, prefix, empty, config['heldout_seeds'], output, step, 'heldout')
            base, initial, trained = (report['conditions'][name] for name in ('prefix', 'initialized', 'trained'))
            passes = (trained['perceptual_diversity'] >= config['minimum_diversity_ratio'] * base['perceptual_diversity']
                      and trained['perceptual_deviation'] <= config['maximum_perceptual_ratio'] * max(initial['perceptual_deviation'], 1e-12)
                      and trained['saturated_pixel_fraction'] <= config['maximum_saturated_pixel_fraction'])
            selected = passes and trained['mean_fake_logit'] < best
            if selected:
                best = trained['mean_fake_logit']
                write_json({'step': step, 'passed_proxy_quality_gates': True, 'visual_review_required': True,
                            'heldout': report, 'not_evidence_of_detector_generalization': True}, output / 'candidate_selection.json')
            if tracker:
                tracker.log({f'heldout/{key}': value for key, value in trained.items() if isinstance(value, (float, int))}, step=step)
        checkpoint = {'step': step, 'prompt': prompt.state_dict(), 'optimizer': optimizer.state_dict(), 'configuration': config,
                      'critic_sha256': identity['sha256'], 'objective': config['objective'], 'best_candidate_logit': best,
                      'early_stopping_counter': stagnant, 'stop_reason': 'prompt_converged' if converged else None}
        atomic_save(checkpoint, output / 'latest.pt')
        if step % config['save_every'] == 0 or step == config['updates'] or converged:
            atomic_save(checkpoint, output / f'prompt_{step:04d}.pt')
        if selected:
            atomic_save(checkpoint, output / 'best_candidate.pt')
        if converged or step == config['updates']:
            write_json({'step': step, 'reason': 'prompt_converged' if converged else 'maximum_updates',
                        'early_stopping_counter': stagnant, **update_stats}, output / 'completion.json')
        if converged:
            print(f'Early stopping at update {step}: prompt updates remained small for {stagnant} consecutive steps.', flush=True)
            break
    if tracker:
        swanlab.finish()


if __name__ == '__main__':
    main()
