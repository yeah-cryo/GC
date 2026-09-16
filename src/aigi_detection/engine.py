import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state()}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state(state['cuda'])


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def write_json(value, path):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def scores(totals):
    loss, correct_real, count_real, correct_fake, count_fake = totals.tolist()
    if not count_real or not count_fake:
        raise ValueError('Evaluation requires both real and fake images')
    real, fake = correct_real / count_real, correct_fake / count_fake
    return {'loss': loss / (count_real + count_fake), 'real_accuracy': real,
            'fake_accuracy': fake, 'balanced_accuracy': (real + fake) / 2,
            'real_count': int(count_real), 'fake_count': int(count_fake)}


@torch.no_grad()
def validate(model, loader, device, use_bf16=True):
    model.eval()
    if dist.is_initialized():
        for buffer in model.buffers():
            dist.broadcast(buffer, src=0)
    totals = torch.zeros(5, dtype=torch.float64, device=device)
    skipped = torch.zeros(2, dtype=torch.int64, device=device)
    for batch in loader:
        crops, labels = batch[:2]
        if len(batch) == 3:
            skipped += torch.tensor([batch[2].count(0), batch[2].count(1)], device=device)
        if crops is None:
            continue
        crops, labels = crops.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        # Process one crop position at a time to bound validation GPU memory.
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            logits = torch.stack([model(crops[:, i]).flatten().float() for i in range(5)]).mean(0)
        loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='sum')
        predicted = logits >= 0
        real, fake = labels == 0, labels == 1
        totals += torch.stack([loss.double(), ((~predicted) & real).sum(), real.sum(),
                               (predicted & fake).sum(), fake.sum()])
    if dist.is_initialized():
        dist.all_reduce(totals)
        dist.all_reduce(skipped)
    return {**scores(totals), 'skipped_real': int(skipped[0].item()), 'skipped_fake': int(skipped[1].item())}
