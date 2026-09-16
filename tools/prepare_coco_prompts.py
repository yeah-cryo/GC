import argparse
import hashlib
import json
from pathlib import Path

from aigi_detection.datasets.coco_prompts import load_coco_captions


def main():
    parser = argparse.ArgumentParser(description='Save ordered COCO 2014 training captions.')
    parser.add_argument('--annotations', required=True,
                        help='captions_train2014.json or the official annotations ZIP')
    parser.add_argument('--output', required=True)
    parser.add_argument('--count', type=int, default=20_000)
    parser.add_argument('--unique-images', action='store_true',
                        help='Keep only the first caption annotation for each distinct image ID.')
    args = parser.parse_args()

    source = Path(args.annotations)
    records = load_coco_captions(source)
    if args.unique_images:
        seen = set()
        unique_records = []
        for record in records:
            if record['image_id'] not in seen:
                seen.add(record['image_id'])
                unique_records.append(record)
        records = unique_records
    if len(records) < args.count:
        raise ValueError(f'Requested {args.count} records, but only {len(records)} are available')
    records = records[:args.count]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as handle:
        for prompt_index, record in enumerate(records):
            record = {**record, 'prompt_index': prompt_index}
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    with source.open('rb') as handle:
        source_sha256 = hashlib.file_digest(handle, 'sha256').hexdigest()
    metadata = {
        'source': str(source.resolve()),
        'source_sha256': source_sha256,
        'split': 'train2014',
        'selection': ('first caption per unique image ID in stored annotation order'
                      if args.unique_images else 'first annotations in stored JSON order'),
        'count': args.count,
        'unique_images': args.unique_images,
    }
    output.with_suffix('.metadata.json').write_text(
        json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
