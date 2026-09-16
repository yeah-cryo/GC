import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from aigi_detection.datasets import ImageDataset, ValidationTransform, official_records
from aigi_detection.datasets.genimage import LABELS, collate_readable
from aigi_detection.engine import seed_everything, seed_worker, validate, write_json
from aigi_detection.models.backbones import build_critic


def main():
    parser = argparse.ArgumentParser(description='Final evaluation only; never used for checkpoint selection.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    device = torch.device(args.device)
    # Load only checkpoints from a trusted source: optimizer/RNG metadata uses pickle.
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint['label_mapping'] != LABELS:
        raise ValueError('Checkpoint label mapping differs from nature=0, ai=1.')
    seed_everything(checkpoint['configuration']['seed'])
    model = build_critic().to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    records = official_records(Path(args.data_root), 'val')
    output = Path(args.output) if args.output else Path(args.checkpoint).parent / 'official_val_metrics.json'
    dataset = ImageDataset(args.data_root, records, ValidationTransform(checkpoint['configuration']['crop_size']),
                           corruption_policy='skip', error_log=output.with_suffix('.corrupt.jsonl'))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=device.type == 'cuda', worker_init_fn=seed_worker, collate_fn=collate_readable)
    result = {'checkpoint': str(Path(args.checkpoint).resolve()), 'epoch': checkpoint.get('epoch'),
              'completed_samples': checkpoint.get('completed_samples'),
              'optimizer_steps': checkpoint.get('optimizer_steps'),
              'data_root': str(Path(args.data_root).resolve()), 'split': 'official_val', 'label_mapping': LABELS,
              **validate(model, loader, device, use_bf16=device.type == 'cuda')}
    output = Path(args.output) if args.output else Path(args.checkpoint).parent / 'official_val_metrics.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(result, output)
    print(result)


if __name__ == '__main__':
    main()
