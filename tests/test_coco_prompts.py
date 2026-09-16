import json
import zipfile

from aigi_detection.datasets.coco_prompts import load_coco_captions, read_prompt_manifest


def test_coco_prompt_order_and_zip_loading(tmp_path):
    payload = {'annotations': [
        {'id': 8, 'image_id': 4, 'caption': ' First caption. '},
        {'id': 3, 'image_id': 4, 'caption': 'Second caption.'},
    ]}
    archive_path = tmp_path / 'annotations.zip'
    with zipfile.ZipFile(archive_path, 'w') as archive:
        archive.writestr('annotations/captions_train2014.json', json.dumps(payload))
    assert load_coco_captions(archive_path) == [
        {'prompt_index': 0, 'annotation_id': 8, 'image_id': 4, 'caption': 'First caption.'},
        {'prompt_index': 1, 'annotation_id': 3, 'image_id': 4, 'caption': 'Second caption.'},
    ]


def test_read_prompt_manifest_limit(tmp_path):
    manifest = tmp_path / 'prompts.jsonl'
    manifest.write_text('{"caption":"one"}\n{"caption":"two"}\n')
    assert read_prompt_manifest(manifest, 1) == [{'caption': 'one'}]
