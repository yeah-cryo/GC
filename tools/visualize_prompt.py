"""Render same-seed comparisons from an existing trained prompt checkpoint."""
import argparse
import json
from pathlib import Path

import torch
from torchvision.utils import save_image

from aigi_detection.engine import seed_everything, write_json
from aigi_detection.losses.losses import critic_logits
from aigi_detection.models.backbones import build_critic
from aigi_detection.models.modules.differentiable_sd import DifferentiableSD
from aigi_detection.models.modules.soft_prompt import SoftPrompt, encode_plain


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    path = Path(args.checkpoint)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    config, step = checkpoint['configuration'], checkpoint['step']
    seed_everything(config['initialization_seed'])
    torch.set_num_threads(2)
    output = path.parent / 'visualizations'
    output.mkdir(exist_ok=True)
    sd = DifferentiableSD(config['model_path'], steps=config['ddim_steps'], guidance=config['guidance_scale'], size=config['resolution'])
    prompt = SoftPrompt(sd.text_encoder, sd.tokenizer, config['prefix'], config['soft_tokens'],
                        config['initialization_seed'], config['initialization_noise_std']).cuda()
    prompt.load_state_dict(checkpoint['prompt'])
    critic = build_critic().cuda().eval().requires_grad_(False)
    critic.load_state_dict(torch.load(path.parent / 'critic.pt', map_location='cpu', weights_only=False)['model'])
    empty = encode_plain(sd.text_encoder, sd.tokenizer, '')
    conditions = {'prefix': encode_plain(sd.text_encoder, sd.tokenizer, config['prefix']),
                  'initialized': prompt(sd.text_encoder, initialized=True), 'trained': prompt(sd.text_encoder)}
    seeds = config['heldout_seeds'][:4]
    images, scores = [], {}
    for name, condition in conditions.items():
        row, scores[name] = [], []
        for seed in seeds:
            image = sd.generate(condition, empty, seed)
            row.append(image.cpu())
            scores[name].append(critic_logits(critic, image).sigmoid().item())
        save_image(torch.cat(row), output / f'{name}_{step:04d}.png', nrow=len(seeds))
        images.extend(row)
    save_image(torch.cat(images), output / f'comparison_{step:04d}.png', nrow=len(seeds))
    result = {'step': step, 'checkpoint': str(path), 'seeds': seeds, 'rows': list(conditions), 'fake_probabilities': scores}
    write_json(result, output / f'comparison_{step:04d}.json')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
