import hashlib
import json
from pathlib import Path

from .genimage import manifest_text


def build_probe_records(metrics_path, generated_root, coco_images_root, expected=20_000,
                        fake_source='guided_sd14'):
    """Pair each guided fake with the COCO image associated with its caption."""
    metrics_path = Path(metrics_path)
    generated_root = Path(generated_root).resolve()
    coco_images_root = Path(coco_images_root).resolve()
    rows = [json.loads(line) for line in metrics_path.read_text(encoding='utf-8').splitlines() if line]
    rows.sort(key=lambda row: row['prompt_index'])
    if len(rows) != expected or [row['prompt_index'] for row in rows] != list(range(expected)):
        raise ValueError(f'Expected exactly {expected} consecutive prompt indices.')
    if not coco_images_root.is_dir() or not generated_root.is_dir():
        raise FileNotFoundError('COCO or generated-image root is missing.')
    records = []
    for row in rows:
        real = coco_images_root / f"COCO_train2014_{row['image_id']:012d}.jpg"
        fake = generated_root / f"strength_{row['strength']:g}_seed_{row['seed']}.png"
        shared = {'prompt_index': row['prompt_index'], 'annotation_id': row['annotation_id'],
                  'image_id': row['image_id'], 'caption': row['prompt']}
        records.append({'path': str(real), 'label': 0, 'source': 'coco2014', **shared})
        records.append({'path': str(fake), 'label': 1, 'source': fake_source,
                        'seed': row['seed'], 'strength': row['strength'], **shared})
    return records


def write_probe_manifest(records, path):
    content = manifest_text(records)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding='utf-8') != content:
        raise ValueError(f'Existing PROBE manifest differs: {path}')
    if not path.exists():
        path.write_text(content, encoding='utf-8')
    return hashlib.sha256(content.encode()).hexdigest()
