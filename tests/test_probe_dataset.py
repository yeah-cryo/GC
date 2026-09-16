import json

from PIL import Image

from aigi_detection.datasets.probe import build_probe_records, write_probe_manifest


def test_probe_records_pair_caption_linked_real_and_guided_fake(tmp_path):
    generated = tmp_path / 'generated'
    coco = tmp_path / 'coco'
    generated.mkdir()
    coco.mkdir()
    rows = []
    for index, image_id in enumerate((7, 7)):
        Image.new('RGB', (4, 4)).save(coco / f'COCO_train2014_{image_id:012d}.jpg')
        Image.new('RGB', (4, 4)).save(generated / f'strength_20_seed_{index}.png')
        rows.append({'strength': 20.0, 'seed': index, 'prompt_index': index,
                     'annotation_id': 100 + index, 'image_id': image_id,
                     'prompt': f'caption {index}'})
    metrics = tmp_path / 'metrics.jsonl'
    metrics.write_text(''.join(json.dumps(row) + '\n' for row in reversed(rows)))
    records = build_probe_records(metrics, generated, coco, expected=2)
    assert [row['label'] for row in records] == [0, 1, 0, 1]
    assert records[0]['image_id'] == records[2]['image_id'] == 7
    assert records[1]['seed'] == 0 and records[3]['seed'] == 1
    digest = write_probe_manifest(records, tmp_path / 'probe.jsonl')
    assert len(digest) == 64
