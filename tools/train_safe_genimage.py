import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T

from aigi_detection.datasets.genimage import (
    LABELS, ImageDataset, build_manifests, collate_readable, read_manifest)
from aigi_detection.engine import restore_rng, rng_state, seed_everything, seed_worker
from aigi_detection.models.backbones.safe_critic import _SAFEBackbone
from pytorch_wavelets import DWTForward


class RandomMask:
    def __init__(self, ratio=(0.0, 0.75), patch_size=16, probability=0.5):
        self.ratio = ratio
        self.patch_size = int(patch_size)
        self.probability = float(probability)

    def __call__(self, tensor):
        if random.random() > self.probability:
            return tensor
        _, height, width = tensor.shape
        ratio = random.uniform(*self.ratio)
        count = int(height * width * ratio / self.patch_size ** 2)
        positions = set()
        while len(positions) < count:
            positions.add((random.randrange(height // self.patch_size) * self.patch_size,
                           random.randrange(width // self.patch_size) * self.patch_size))
        mask = torch.ones((height, width), dtype=tensor.dtype)
        for top, left in positions:
            mask[top:top + self.patch_size, left:left + self.patch_size] = 0
        return tensor * mask.unsqueeze(0)


class SAFETrainTransform:
    def __init__(self, size=256):
        self.transform = T.Compose([
            T.RandomCrop((size, size), pad_if_needed=True),
            T.RandomHorizontalFlip(0.5),
            T.RandomRotation(180),
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            T.ToTensor(),
            RandomMask((0.0, 0.75), 16, 0.5),
        ])

    def __call__(self, image, label):
        return self.transform(image)


class SAFEEvalTransform:
    def __init__(self, size=256):
        self.transform = T.Compose([T.CenterCrop((size, size)), T.ToTensor()])

    def __call__(self, image, label):
        return self.transform(image)


class SAFEModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _SAFEBackbone()
        for module in self.backbone.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, torch.nn.BatchNorm2d):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)
        self.wavelet = DWTForward(J=1, mode='symmetric', wave='bior1.3')

    def forward(self, images):
        _, high = self.wavelet(images.float())
        diagonal = high[0][:, :, 2]
        diagonal = F.interpolate(diagonal, images.shape[-2:], mode='bilinear',
                                 align_corners=False, antialias=True)
        return self.backbone(diagonal)


def learning_rate(step, steps_per_epoch, config):
    epoch = step / steps_per_epoch
    if epoch < config['warmup_epochs']:
        return config['learning_rate'] * epoch / config['warmup_epochs']
    progress = (epoch - config['warmup_epochs']) / (config['epochs'] - config['warmup_epochs'])
    return config['min_learning_rate'] + (config['learning_rate'] - config['min_learning_rate']) * 0.5 * (1 + math.cos(math.pi * progress))


def save_checkpoint(path, model, optimizer, epoch, best_score, config, split_hash,
                    data_generator):
    payload = {
        'model': model.backbone.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'best_score': best_score,
        'config': config,
        'label_mapping': LABELS,
        'split_sha256': split_hash,
        'architecture': 'SAFE bior1.3 diagonal-wavelet truncated ResNet-50',
        'rng_states': rng_state(),
        'data_generator_state': data_generator.get_state(),
    }
    temporary = Path(str(path) + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def validate(model, loader, device, max_batches=None):
    model.eval()
    loss_sum = 0.0
    counts = torch.zeros(2, dtype=torch.long)
    correct = torch.zeros(2, dtype=torch.long)
    seen = 0
    skipped = 0
    for index, (images, labels, bad_labels) in enumerate(loader):
        skipped += len(bad_labels)
        if images is None:
            continue
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(images)
            loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
        predictions = logits.argmax(1)
        loss_sum += loss.item() * len(labels)
        seen += len(labels)
        for label in (0, 1):
            selected = labels == label
            counts[label] += selected.sum().cpu()
            correct[label] += (predictions[selected] == label).sum().cpu()
        if max_batches is not None and index + 1 >= max_batches:
            break
    accuracies = correct.float() / counts.clamp_min(1)
    return {'loss': loss_sum / seen, 'real_accuracy': accuracies[0].item(),
            'fake_accuracy': accuracies[1].item(),
            'balanced_accuracy': accuracies.mean().item(), 'images': seen,
            'skipped_corrupt': skipped}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/experiments/safe_sd14_genimage.yaml')
    parser.add_argument('--resume')
    parser.add_argument('--output')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--global-batch', type=int)
    parser.add_argument('--micro-batch', type=int)
    parser.add_argument('--max-train-batches', type=int)
    parser.add_argument('--max-validation-batches', type=int)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    for key, value in [('output', args.output), ('epochs', args.epochs),
                       ('global_batch_size', args.global_batch),
                       ('micro_batch_size', args.micro_batch)]:
        if value is not None:
            config[key] = value
    if config['global_batch_size'] % config['micro_batch_size']:
        raise ValueError('Global batch size must be divisible by microbatch size.')
    if config['warmup_epochs'] >= config['epochs']:
        raise ValueError('Warmup epochs must be smaller than total epochs.')

    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'configuration.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    seed_everything(config['seed'])
    device = torch.device('cuda')

    build_manifests(config['data_root'], output / 'manifests', config['seed'],
                    config['validation_fraction'], config['expected_per_class'])
    split = json.loads((output / 'manifests/split.json').read_text())
    train_records = read_manifest(output / 'manifests/train.jsonl')
    validation_records = read_manifest(output / 'manifests/validation.jsonl')
    train_set = ImageDataset(config['data_root'], train_records,
                             SAFETrainTransform(config['input_size']), 'replace',
                             output / 'corrupt_train.jsonl')
    validation_set = ImageDataset(config['data_root'], validation_records,
                                  SAFEEvalTransform(config['input_size']), 'skip',
                                  output / 'corrupt_validation.jsonl')
    generator = torch.Generator().manual_seed(config['seed'])
    train_loader = DataLoader(
        train_set, batch_size=config['global_batch_size'], shuffle=True,
        generator=generator, num_workers=config['workers'], pin_memory=True,
        drop_last=True, persistent_workers=config['workers'] > 0,
        worker_init_fn=seed_worker)
    validation_loader = DataLoader(
        validation_set, batch_size=config['validation_batch_size'], shuffle=False,
        num_workers=config['workers'], pin_memory=True, collate_fn=collate_readable,
        persistent_workers=config['workers'] > 0, worker_init_fn=seed_worker)

    model = SAFEModel().to(device)
    optimizer = torch.optim.AdamW(model.backbone.parameters(), lr=config['learning_rate'],
                                  betas=(0.9, 0.999), eps=1e-8,
                                  weight_decay=config['weight_decay'])
    start_epoch, best_score = 0, -1.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        if checkpoint['label_mapping'] != LABELS or checkpoint['split_sha256'] != split['sha256']:
            raise ValueError('Resume checkpoint label mapping or split differs.')
        model.backbone.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        generator.set_state(checkpoint['data_generator_state'])
        restore_rng(checkpoint['rng_states'])
        start_epoch = checkpoint['epoch'] + 1
        best_score = checkpoint['best_score']

    steps_per_epoch = len(train_loader)
    accumulation = config['global_batch_size'] // config['micro_batch_size']
    metrics_path = output / 'metrics.jsonl'
    print(json.dumps({'training_images': len(train_set),
                      'validation_images': len(validation_set),
                      'optimizer_steps_per_epoch': steps_per_epoch,
                      'global_batch_size': config['global_batch_size'],
                      'micro_batch_size': config['micro_batch_size'],
                      'accumulation_steps': accumulation,
                      'parameters': sum(p.numel() for p in model.backbone.parameters())}), flush=True)

    for epoch in range(start_epoch, config['epochs']):
        model.train()
        started = time.monotonic()
        loss_sum, seen = 0.0, 0
        for batch_index, (images, labels) in enumerate(train_loader):
            step = epoch * steps_per_epoch + batch_index
            lr = learning_rate(step, steps_per_epoch, config)
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0, len(labels), config['micro_batch_size']):
                micro_images = images[offset:offset + config['micro_batch_size']].to(device, non_blocking=True)
                micro_labels = labels[offset:offset + config['micro_batch_size']].to(device, non_blocking=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(micro_images)
                    loss = F.cross_entropy(logits, micro_labels, label_smoothing=0.1)
                (loss / accumulation).backward()
                loss_sum += loss.item() * len(micro_labels)
                seen += len(micro_labels)
            optimizer.step()
            if (batch_index + 1) % config['print_frequency'] == 0:
                print(json.dumps({'epoch': epoch + 1, 'batch': batch_index + 1,
                                  'batches': steps_per_epoch, 'loss': loss_sum / seen,
                                  'lr': lr, 'seconds': time.monotonic() - started}), flush=True)
            if args.max_train_batches is not None and batch_index + 1 >= args.max_train_batches:
                break

        validation = validate(model, validation_loader, device, args.max_validation_batches)
        row = {'epoch': epoch + 1, 'train_loss': loss_sum / seen,
               'learning_rate': optimizer.param_groups[0]['lr'],
               'seconds': time.monotonic() - started, **validation}
        with metrics_path.open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        improved = validation['balanced_accuracy'] > best_score + 0.0001
        if improved:
            best_score = validation['balanced_accuracy']
            save_checkpoint(output / 'best.pt', model, optimizer, epoch, best_score,
                            config, split['sha256'], generator)
        save_checkpoint(output / 'latest.pt', model, optimizer, epoch, best_score,
                        config, split['sha256'], generator)
        print(json.dumps({**row, 'best_balanced_accuracy': best_score,
                          'saved_best': improved}), flush=True)


if __name__ == '__main__':
    main()
