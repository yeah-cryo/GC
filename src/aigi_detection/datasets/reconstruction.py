from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class ReconstructionPairDataset(Dataset):
    """Deterministic real/fake pairs with replacement for unreadable files."""

    def __init__(self, root, real_records, fake_records, size=512, max_attempts=32):
        if len(real_records) != len(fake_records) or not real_records:
            raise ValueError('Reconstruction requires equally sized nonempty classes.')
        self.root = Path(root)
        self.real_records, self.fake_records = real_records, fake_records
        self.max_attempts = int(max_attempts)
        self.transform = T.Compose((
            T.Resize(size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(size), T.ToTensor(),
        ))

    def __len__(self):
        return len(self.real_records)

    def _load(self, record):
        with Image.open(self.root / record['path']) as image:
            return self.transform(image.convert('RGB'))

    def __getitem__(self, index):
        errors = []
        for offset in range(min(self.max_attempts, len(self))):
            candidate = (index + offset) % len(self)
            try:
                return {
                    'real': self._load(self.real_records[candidate]),
                    'fake': self._load(self.fake_records[candidate]),
                    'requested_index': index,
                    'used_index': candidate,
                }
            except (OSError, ValueError, SyntaxError) as error:
                errors.append(f'{type(error).__name__}: {error}')
        raise RuntimeError(f'No readable reconstruction pair near index {index}: {errors[-1]}')
