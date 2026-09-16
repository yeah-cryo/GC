"""Score saved classifier-guided images with a ResNet critic."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor

from aigi_detection.losses.losses import critic_logits
from aigi_detection.models.backbones import build_critic


class SavedImages(Dataset):
    def __init__(self, root, rows):
        self.root, self.rows = Path(root), rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.root / f"strength_{row['strength']:g}_seed_{row['seed']}.png"
        with Image.open(path) as image:
            return to_tensor(image.convert('RGB')), index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--metrics', required=True)
    parser.add_argument('--images-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.metrics).read_text(encoding='utf-8').splitlines() if line]
    rows.sort(key=lambda row: row['prompt_index'])
    if not rows or len({row['prompt_index'] for row in rows}) != len(rows):
        raise ValueError('Metrics must contain unique prompt indices.')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint['label_mapping'] != {'nature': 0, 'ai': 1}:
        raise ValueError('Expected nature=0 and ai=1.')
    model = build_critic().cuda().eval().requires_grad_(False)
    model.load_state_dict(checkpoint['model'], strict=True)
    loader = DataLoader(SavedImages(args.images_root, rows), batch_size=args.batch_size,
                        num_workers=args.workers, shuffle=False, pin_memory=True)

    results, probability_sum, detected = [], 0.0, 0
    with torch.no_grad():
        for batch_index, (images, indices) in enumerate(loader, start=1):
            logits = critic_logits(model, images.cuda(non_blocking=True)).cpu()
            probabilities = logits.sigmoid()
            probability_sum += probabilities.double().sum().item()
            detected += (logits >= 0).sum().item()
            for index, logit, probability in zip(indices.tolist(), logits.tolist(), probabilities.tolist()):
                source = rows[index]
                results.append({'prompt_index': source['prompt_index'], 'seed': source['seed'],
                                'strength': source['strength'], 'fake_logit': logit,
                                'fake_probability': probability, 'classified_fake': logit >= 0})
            if batch_index == 1 or batch_index % 50 == 0:
                print(f'scored={len(results)}/{len(rows)}', flush=True)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'scores.jsonl').open('w', encoding='utf-8') as handle:
        for row in results:
            handle.write(json.dumps(row) + '\n')
    with Path(args.checkpoint).open('rb') as handle:
        checkpoint_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
    summary = {
        'checkpoint': str(Path(args.checkpoint).resolve()), 'checkpoint_sha256': checkpoint_hash,
        'checkpoint_epoch': checkpoint['epoch'], 'images': len(results),
        'guidance_strengths': sorted({row['strength'] for row in rows}),
        'mean_fake_probability': probability_sum / len(results),
        'classified_fake_count': detected, 'classified_fake_rate': detected / len(results),
        'classified_real_count': len(results) - detected,
        'classified_real_rate': 1 - detected / len(results),
        'original_critic_mean_fake_probability': sum(row['saved_png_fake_probability'] for row in rows) / len(rows),
        'protocol': 'saved RGB PNG; five fixed 224x224 crops; ImageNet normalization; mean logits; FP32',
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
