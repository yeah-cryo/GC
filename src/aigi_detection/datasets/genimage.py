import hashlib
import json
import os
import random
import warnings
import fcntl
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import default_collate

LABELS = {'nature': 0, 'ai': 1}
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}


def official_records(root, split):
    records = []
    for name, label in LABELS.items():
        directory = Path(root) / split / name
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        # os.walk reuses directory-entry types instead of stat-ing every file.
        # Per-file metadata requests are particularly costly on mounted Windows drives.
        paths = sorted((Path(parent) / name).relative_to(root).as_posix()
                       for parent, _, names in os.walk(directory)
                       for name in names if Path(name).suffix.lower() in EXTENSIONS)
        if not paths:
            raise ValueError(f'No images in {directory}')
        records.extend({'path': p, 'label': label} for p in paths)
    return records


def manifest_text(records):
    return ''.join(json.dumps(record, sort_keys=True) + '\n' for record in records)


def read_manifest(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_manifests(root, output, seed=42, fraction=0.05, expected_per_class=162000):
    """One shared RNG; sample validation indices from sorted real, then fake paths."""
    records = official_records(Path(root), 'train')
    rng = random.Random(seed)
    train, validation = [], []
    for label in LABELS.values():
        group = [r for r in records if r['label'] == label]
        if len(group) != expected_per_class:
            raise ValueError(f'Class {label}: expected {expected_per_class} images, found {len(group)}')
        held_out = set(rng.sample(range(len(group)), round(len(group) * fraction)))
        train.extend(r for i, r in enumerate(group) if i not in held_out)
        validation.extend(r for i, r in enumerate(group) if i in held_out)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, rows in [('train', train), ('validation', validation)]:
        content = manifest_text(rows)
        path = output / f'{name}.jsonl'
        if path.exists() and path.read_text() != content:
            raise ValueError(f'Existing manifest differs from reconstructed split: {path}')
        if not path.exists():
            path.write_text(content)
        hashes[name] = hashlib.sha256(content.encode()).hexdigest()
    metadata = {'seed': seed, 'validation_fraction': fraction, 'algorithm': 'sorted paths; shared random.Random.sample; nature then ai',
                'labels': LABELS, 'training_images': len(train), 'validation_images': len(validation), 'sha256': hashes}
    (output / 'split.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return hashes


class ImageDataset(Dataset):
    def __init__(self, root, records, transform, corruption_policy='raise', error_log=None, max_attempts=32):
        self.root, self.records, self.transform = Path(root), records, transform
        if corruption_policy not in ('raise', 'replace', 'skip') or max_attempts < 1:
            raise ValueError('Invalid corruption policy or retry limit')
        self.corruption_policy, self.error_log, self.max_attempts = corruption_policy, error_log, max_attempts
        self.bad_indices = set()
        self.class_indices, self.class_positions = {}, {}
        if corruption_policy == 'replace':
            for index, row in enumerate(records):
                group = self.class_indices.setdefault(row['label'], [])
                self.class_positions[index] = len(group)
                group.append(index)
        if error_log:
            Path(error_log).parent.mkdir(parents=True, exist_ok=True)

    def _decode(self, index):
        row = self.records[index]
        try:
            with Image.open(self.root / row['path']) as image:
                # Force decoding inside the handler; augmentation errors must propagate.
                return image.convert('RGB')
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
            if self.corruption_policy == 'raise':
                raise
            self.bad_indices.add(index)
            event = {'path': row['path'], 'label': row['label'], 'policy': self.corruption_policy,
                     'error': type(error).__name__, 'message': str(error)}
            if self.error_log:
                # Workers/ranks share this append-only audit log.
                with open(self.error_log, 'a') as handle:
                    fcntl.flock(handle, fcntl.LOCK_EX)
                    handle.write(json.dumps(event) + '\n')
                    handle.flush()
                    fcntl.flock(handle, fcntl.LOCK_UN)
            warnings.warn(f"Unreadable image ({self.corruption_policy}): {row['path']}: {error}", RuntimeWarning)
            return None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        if self.corruption_policy == 'replace':
            group = self.class_indices[row['label']]
            start = self.class_positions[index]
            candidates = (group[(start + offset) % len(group)] for offset in range(min(self.max_attempts, len(group))))
        else:
            candidates = (index,)
        for candidate in candidates:
            if candidate in self.bad_indices:
                continue
            image = self._decode(candidate)
            if image is not None:
                return self.transform(image, row['label']), row['label']
        if self.corruption_policy == 'skip':
            return None, row['label']
        raise RuntimeError(f"No readable class-{row['label']} image within {self.max_attempts} attempts starting at {row['path']}")


def collate_readable(batch):
    """Validation keeps genuine samples only, including fully unreadable batches."""
    valid = [(image, label) for image, label in batch if image is not None]
    skipped = [label for image, label in batch if image is None]
    if not valid:
        return None, None, skipped
    images, labels = default_collate(valid)
    return images, labels, skipped
