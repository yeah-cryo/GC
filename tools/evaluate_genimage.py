"""Evaluate the selected critic on every non-SD1.4 GenImage validation split."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from aigi_detection.datasets import ImageDataset, ValidationTransform, official_records
from aigi_detection.datasets.genimage import LABELS, collate_readable
from aigi_detection.engine import seed_everything, seed_worker, validate, write_json
from aigi_detection.models.backbones import build_critic

GENERATORS = ('ADM', 'BigGAN', 'Midjourney', 'VQDM', 'glide', 'stable_diffusion_v_1_5', 'wukong')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='outputs/resnet50_critic/best.pt')
    parser.add_argument('--data-root', default='/mnt/f/datasets/GenImage')
    parser.add_argument('--output', default='outputs/resnet50_critic/genimage_evaluation')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    roots = {}
    for name in GENERATORS:
        candidates = [p for p in (Path(args.data_root) / name).iterdir()
                      if (p / 'val/nature').is_dir() and (p / 'val/ai').is_dir()]
        if len(candidates) != 1:
            raise ValueError(f'Expected one official validation root for {name}: {candidates}')
        roots[name] = candidates[0]
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint['label_mapping'] != LABELS:
        raise ValueError('Unexpected checkpoint label mapping')
    seed_everything(checkpoint['configuration']['seed'])
    device = torch.device('cuda')
    model = build_critic().to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    with open(args.checkpoint, 'rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    metadata = {'checkpoint': str(Path(args.checkpoint).resolve()), 'checkpoint_sha256': digest,
                'epoch': checkpoint['epoch'], 'label_mapping': LABELS, 'split': 'official_val',
                'protocol': 'fake JPEG quality 96; zero padding; five 224 crops; ImageNet normalization; mean logits >= 0',
                'precision': 'bf16', 'expected_generators': list(GENERATORS)}
    write_json(metadata, output / 'configuration.json')
    results = []
    for name, root in roots.items():
        print(f'Evaluating {name}...', flush=True)
        started = time.monotonic()
        records = official_records(root, 'val')
        dataset = ImageDataset(root, records, ValidationTransform(checkpoint['configuration']['crop_size']),
                               corruption_policy='skip', error_log=output / f'{name}.corrupt.jsonl')
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False,
                            pin_memory=True, worker_init_fn=seed_worker, collate_fn=collate_readable)
        result = {'generator': name, 'data_root': str(root), 'manifest_images': len(records),
                  **validate(model, loader, device), 'seconds': time.monotonic() - started}
        write_json({**metadata, **result}, output / f'{name}.json')
        results.append(result)
        summary = {'metadata': metadata, 'complete': len(results) == len(GENERATORS), 'results': results,
                   'macro_balanced_accuracy': sum(row['balanced_accuracy'] for row in results) / len(results)}
        write_json(summary, output / 'summary.json')
        with (output / 'summary.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result))
            writer.writeheader()
            writer.writerows(results)
        print(json.dumps(result), flush=True)
    print(f'Complete: macro balanced accuracy={summary["macro_balanced_accuracy"]:.6f}', flush=True)


if __name__ == '__main__':
    main()
