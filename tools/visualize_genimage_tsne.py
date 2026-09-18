"""Visualize GenImage validation embeddings from a trained ResNet-50 critic."""
import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset

from aigi_detection.datasets import ValidationTransform, official_records
from aigi_detection.models.backbones import build_critic


LABEL_NAMES = {0: 'real', 1: 'fake'}


def generator_root(data_root, directory):
    candidates = [path for path in (Path(data_root) / directory).iterdir()
                  if (path / 'val/nature').is_dir() and (path / 'val/ai').is_dir()]
    if len(candidates) != 1:
        raise ValueError(f'Expected one official validation root under {directory}: {candidates}')
    return candidates[0]


def sample_records(config):
    per_source = int(config['samples_per_source'])
    rng = random.Random(config['seed'])
    sampled = []
    roots = {generator: generator_root(config['data_root'], directory)
             for generator, directory in config['generators'].items()}
    real_source = config['real_source']
    real_records = sorted((row for row in official_records(roots[real_source], 'val')
                           if row['label'] == 0), key=lambda row: row['path'])
    if len(real_records) < per_source:
        raise ValueError(f'Real source has only {len(real_records)} images.')
    sampled.extend({'generator': 'Real', 'root': str(roots[real_source]), **row}
                   for row in rng.sample(real_records, per_source))
    for generator, directory in config['generators'].items():
        root = roots[generator]
        records = official_records(root, 'val')
        group = sorted((row for row in records if row['label'] == 1),
                       key=lambda row: row['path'])
        if len(group) < per_source:
            raise ValueError(f'{generator} has only {len(group)} fake images.')
        sampled.extend({'generator': generator, 'root': str(root), **row}
                       for row in rng.sample(group, per_source))
    return sampled


class EmbeddingDataset(Dataset):
    def __init__(self, records, crop_size):
        self.records = records
        self.transform = ValidationTransform(crop_size)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from PIL import Image
        row = self.records[index]
        try:
            with Image.open(Path(row['root']) / row['path']) as image:
                crops = self.transform(image.convert('RGB'), row['label'])
            return crops, row['label'], index, ''
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
            return None, row['label'], index, f'{type(error).__name__}: {error}'


def collate(batch):
    readable = [row for row in batch if row[0] is not None]
    failures = [{'index': row[2], 'error': row[3]} for row in batch if row[0] is None]
    if not readable:
        return None, None, None, failures
    crops = torch.stack([row[0] for row in readable])
    labels = torch.tensor([row[1] for row in readable])
    indices = torch.tensor([row[2] for row in readable])
    return crops, labels, indices, failures


def plot(coordinates, rows, output):
    generators = list(dict.fromkeys(row['generator'] for row in rows))
    palette = plt.get_cmap('tab10')
    figure, axes = plt.subplots(1, 2, figsize=(15, 6.4), constrained_layout=True)
    for number, generator in enumerate(generators):
        mask = np.array([row['generator'] == generator for row in rows])
        axes[0].scatter(coordinates[mask, 0], coordinates[mask, 1], s=8, alpha=.58,
                        linewidths=0, color=palette(number), label=generator)
    axes[0].legend(markerscale=2.5, frameon=False, ncol=2)
    axes[0].set_title('Color: image source')
    for label, color, marker in ((0, '#2878B5', 'o'), (1, '#D95319', 'x')):
        mask = np.array([row['label'] == label for row in rows])
        axes[1].scatter(coordinates[mask, 0], coordinates[mask, 1], s=9, alpha=.48,
                        linewidths=.45 if label else 0, color=color, marker=marker,
                        label=LABEL_NAMES[label])
    axes[1].legend(markerscale=2.5, frameon=False)
    axes[1].set_title('Color: ground-truth class')
    for axis in axes:
        axis.set_xlabel('t-SNE 1')
        axis.set_ylabel('t-SNE 2')
        axis.set_xticks([])
        axis.set_yticks([])
        axis.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    figure.suptitle('GenImage validation embeddings — ResNet-50 online-guided 40K + cosine LR',
                    fontsize=14)
    figure.savefig(output / 'genimage_tsne.png', dpi=300, facecolor='white')
    figure.savefig(output / 'genimage_tsne.pdf', bbox_inches='tight', facecolor='white')
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments/resnet50_online40k_genimage_tsne.yaml')
    parser.add_argument('--output')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.output:
        config['output'] = args.output
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for embedding extraction.')
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    records = sample_records(config)
    manifest = ''.join(json.dumps(row, sort_keys=True) + '\n' for row in records)
    (output / 'manifest.jsonl').write_text(manifest)
    (output / 'configuration.yaml').write_text(yaml.safe_dump(config, sort_keys=False))

    checkpoint = torch.load(config['checkpoint'], map_location='cpu', weights_only=False)
    if checkpoint.get('label_mapping') != {'nature': 0, 'ai': 1}:
        raise ValueError('Unexpected checkpoint label mapping.')
    device = torch.device('cuda')
    model = build_critic().to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval().requires_grad_(False)
    del checkpoint
    feature_model = torch.nn.Sequential(*list(model.children())[:-1])
    dataset = EmbeddingDataset(records, config['crop_size'])
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False,
                        num_workers=config['workers'], pin_memory=True,
                        persistent_workers=config['workers'] > 0, collate_fn=collate)
    embeddings, logits, kept, failures = [], [], [], []
    started = time.monotonic()
    with torch.inference_mode():
        for batch_number, (crops, labels, indices, batch_failures) in enumerate(loader, 1):
            failures.extend(batch_failures)
            if crops is None:
                continue
            batch, views = crops.shape[:2]
            flat = crops.flatten(0, 1).to(device, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                features = feature_model(flat).flatten(1)
            features = features.float().reshape(batch, views, -1).mean(1)
            batch_logits = model.fc(features).flatten()
            embeddings.append(features.cpu())
            logits.append(batch_logits.cpu())
            kept.extend(indices.tolist())
            print(json.dumps({'batches': batch_number, 'embedded': len(kept),
                              'total': len(records)}), flush=True)
    matrix = torch.cat(embeddings).numpy()
    scores = torch.cat(logits).sigmoid().numpy()
    kept_rows = [records[index] for index in kept]
    np.save(output / 'embeddings.npy', matrix)
    pca = PCA(n_components=config['pca_components'], random_state=config['seed'])
    reduced = pca.fit_transform(matrix)
    tsne = TSNE(n_components=2, perplexity=config['tsne_perplexity'],
                learning_rate=config['tsne_learning_rate'], max_iter=config['tsne_iterations'],
                init='pca', random_state=config['seed'], method='barnes_hut', verbose=1)
    coordinates = tsne.fit_transform(reduced)
    fields = ('generator', 'label', 'class_name', 'fake_probability', 'tsne_x', 'tsne_y',
              'root', 'path')
    with (output / 'tsne_coordinates.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, probability, point in zip(kept_rows, scores, coordinates):
            writer.writerow({**row, 'class_name': LABEL_NAMES[row['label']],
                             'fake_probability': float(probability),
                             'tsne_x': float(point[0]), 'tsne_y': float(point[1])})
    plot(coordinates, kept_rows, output)
    with open(config['checkpoint'], 'rb') as handle:
        checkpoint_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
    counts = {source: sum(row['generator'] == source for row in kept_rows)
              for source in ('Real', *config['generators'])}
    summary = {
        'images': len(kept_rows), 'embedding_dimensions': int(matrix.shape[1]),
        'pca_components': config['pca_components'],
        'pca_explained_variance_ratio': float(pca.explained_variance_ratio_.sum()),
        'tsne_kl_divergence': float(tsne.kl_divergence_), 'counts': counts,
        'failures': failures, 'seconds': time.monotonic() - started,
        'manifest_sha256': hashlib.sha256(manifest.encode()).hexdigest(),
        'checkpoint': str(Path(config['checkpoint']).resolve()),
        'checkpoint_sha256': checkpoint_hash,
        'feature': 'mean penultimate ResNet-50 embedding over official five validation crops',
        'preprocessing': 'fake JPEG quality 96; zero-pad; five 224 crops; ImageNet normalization',
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
