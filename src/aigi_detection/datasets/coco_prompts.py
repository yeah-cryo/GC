import json
import zipfile
from pathlib import Path


def load_coco_captions(source):
    """Load COCO captions without changing their annotation-file order."""
    source = Path(source)
    if source.suffix.lower() == '.zip':
        with zipfile.ZipFile(source) as archive:
            with archive.open('annotations/captions_train2014.json') as handle:
                payload = json.load(handle)
    else:
        with source.open('rb') as handle:
            payload = json.load(handle)
    return [
        {
            'prompt_index': index,
            'annotation_id': annotation['id'],
            'image_id': annotation['image_id'],
            'caption': annotation['caption'].strip(),
        }
        for index, annotation in enumerate(payload['annotations'])
    ]


def read_prompt_manifest(path, limit=None):
    records = []
    with Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
                if limit is not None and len(records) == limit:
                    break
    return records
