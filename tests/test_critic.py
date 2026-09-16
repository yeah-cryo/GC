import random
import json

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from aigi_detection.datasets import TrainTransform, ValidationTransform, build_manifests, read_manifest
from aigi_detection.datasets.genimage import ImageDataset, collate_readable
from aigi_detection.datasets.transforms import jpeg, normalize, pad
from aigi_detection.engine import atomic_save, scores, validate
from aigi_detection.models.backbones import build_critic
from torchvision.transforms import functional as F


def test_split_shared_rng_and_no_official_val(tmp_path):
    for category in ('nature', 'ai'):
        directory = tmp_path / 'data/train' / category
        directory.mkdir(parents=True)
        for i in range(100):
            (directory / f'{i:03d}.jpg').touch()
    output = tmp_path / 'manifests'
    hashes = build_manifests(tmp_path / 'data', output, expected_per_class=100)
    assert hashes == build_manifests(tmp_path / 'data', output, expected_per_class=100)
    train, val = read_manifest(output / 'train.jsonl'), read_manifest(output / 'validation.jsonl')
    assert len(train) == 190 and len(val) == 10
    assert not {r['path'] for r in train} & {r['path'] for r in val}
    rng = random.Random(42)
    expected = []
    for name in ('nature', 'ai'):
        expected.extend(f'train/{name}/{i:03d}.jpg' for i in sorted(rng.sample(range(100), 5)))
    assert [r['path'] for r in val] == expected
    (output / 'train.jsonl').write_text('changed')
    with pytest.raises(ValueError, match='differs'):
        build_manifests(tmp_path / 'data', output, expected_per_class=100)


def test_validation_crops_and_fake_only_jpeg():
    pixels = np.random.default_rng(10).integers(0, 256, (180, 300, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    transform = ValidationTransform()
    for label in (0, 1):
        processed = jpeg(image, 96) if label else image
        expected = torch.stack([normalize(c) for c in F.five_crop(pad(processed), 224)])
        torch.testing.assert_close(transform(image, label), expected)
    assert not torch.equal(transform(image, 0), transform(image, 1))


def test_augment_small_image():
    image = Image.new('RGB', (70, 80), (110, 120, 130))
    for seed in range(12):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        result = TrainTransform()(image, 0)
        assert result.shape == (3, 224, 224) and torch.isfinite(result).all()


def test_validation_averages_logits_and_zero_is_fake():
    class ReadLogit(torch.nn.Module):
        def forward(self, x):
            return x[:, :1, 0, 0]
    # Mean logits are 1, 0, -1. Sigmoid-before-average would misclassify first image.
    crops = torch.tensor([[9., -1., -1., -1., -1.], [0., 0., 0., 0., 0.], [-1., -1., -1., -1., -1.]]).reshape(3, 5, 1, 1, 1)
    labels = torch.tensor([1, 0, 0])
    result = validate(ReadLogit(), DataLoader(TensorDataset(crops, labels), batch_size=2), torch.device('cpu'), False)
    assert result['fake_accuracy'] == 1 and result['real_accuracy'] == 0.5
    assert result['balanced_accuracy'] == 0.75
    expected = torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor([1., 0., -1.]), labels.float()).item()
    assert result['loss'] == pytest.approx(expected)


def test_balanced_accuracy_is_not_overall_accuracy():
    assert scores(torch.tensor([2., 1., 2., 9., 9.]))['balanced_accuracy'] == 0.75


def test_checkpoint_and_input_gradients(tmp_path):
    model = build_critic('models/resnet50-11ad3fa6.pth')
    assert model.fc.out_features == 1 and all(p.requires_grad for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    backbone_before, head_before = model.conv1.weight.detach().clone(), model.fc.weight.detach().clone()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(torch.randn(2, 3, 32, 32)).flatten(), torch.tensor([0., 1.]))
    loss.backward()
    optimizer.step()
    assert not torch.equal(backbone_before, model.conv1.weight)
    assert not torch.equal(head_before, model.fc.weight)
    atomic_save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, tmp_path / 'critic.pt')
    loaded = build_critic()
    checkpoint = torch.load(tmp_path / 'critic.pt', weights_only=True)
    loaded.load_state_dict(checkpoint['model'])
    resumed_optimizer = torch.optim.AdamW(loaded.parameters(), lr=1e-4)
    resumed_optimizer.load_state_dict(checkpoint['optimizer'])
    assert len(resumed_optimizer.state) == len(optimizer.state)
    assert resumed_optimizer.state[loaded.conv1.weight]['step'].item() == 1
    torch.testing.assert_close(resumed_optimizer.state[loaded.conv1.weight]['exp_avg'], optimizer.state[model.conv1.weight]['exp_avg'])
    loaded.eval().requires_grad_(False)
    image = torch.randn(1, 3, 32, 32, requires_grad=True)
    loaded(image).sum().backward()
    assert image.grad is not None and image.grad.abs().sum() > 0


def test_corrupt_training_replaces_with_same_class_and_logs(tmp_path):
    (tmp_path / 'bad.png').write_bytes(b'not an image')
    Image.new('RGB', (20, 20), (255, 0, 0)).save(tmp_path / 'real.png')
    Image.new('RGB', (20, 20), (0, 255, 0)).save(tmp_path / 'fake.png')
    records = [{'path': 'bad.png', 'label': 1}, {'path': 'real.png', 'label': 0}, {'path': 'fake.png', 'label': 1}]
    dataset = ImageDataset(tmp_path, records, lambda image, label: image.getpixel((0, 0)),
                           corruption_policy='replace', error_log=tmp_path / 'errors.jsonl')
    with pytest.warns(RuntimeWarning, match='bad.png'):
        assert dataset[0] == ((0, 255, 0), 1)
    assert dataset[0] == ((0, 255, 0), 1)
    events = [json.loads(line) for line in (tmp_path / 'errors.jsonl').read_text().splitlines()]
    assert len(events) == 1 and events[0]['path'] == 'bad.png'
    assert events[0]['error'] == 'UnidentifiedImageError'


def test_corrupt_training_retry_is_bounded_and_transform_errors_propagate(tmp_path):
    records = [{'path': 'missing.png', 'label': 0}]
    dataset = ImageDataset(tmp_path, records, TrainTransform(), corruption_policy='replace')
    with pytest.warns(RuntimeWarning), pytest.raises(RuntimeError, match='No readable class-0'):
        dataset[0]
    Image.new('RGB', (20, 20)).save(tmp_path / 'missing.png')
    def broken_transform(image, label):
        raise ValueError('augmentation bug')
    dataset = ImageDataset(tmp_path, records, broken_transform, corruption_policy='replace')
    with pytest.raises(ValueError, match='augmentation bug'):
        dataset[0]


def test_corrupt_validation_skips_without_inflating_metrics(tmp_path):
    for name in ('real.png', 'fake.png'):
        Image.new('RGB', (20, 20)).save(tmp_path / name)
    records = [{'path': 'bad0.png', 'label': 0}, {'path': 'bad1.png', 'label': 1},
               {'path': 'real.png', 'label': 0}, {'path': 'fake.png', 'label': 1}]
    dataset = ImageDataset(tmp_path, records, lambda image, label: torch.full((5, 1, 1, 1), 2. * label - 1),
                           corruption_policy='skip')
    class ReadLogit(torch.nn.Module):
        def forward(self, x):
            return x[:, :1, 0, 0]
    # First batch is entirely unreadable; evaluation must still reach second batch.
    with pytest.warns(RuntimeWarning):
        result = validate(ReadLogit(), DataLoader(dataset, batch_size=2, collate_fn=collate_readable), torch.device('cpu'), False)
    assert result['real_count'] == result['fake_count'] == 1
    assert result['skipped_real'] == result['skipped_fake'] == 1
    assert result['balanced_accuracy'] == 1.0
