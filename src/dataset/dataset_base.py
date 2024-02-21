import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from src.utils.parameters import CColorType
from src.reader.reader_base import CA2D2Reader


class CDatasetBase(Dataset):

    def __init__(self, f_reader: CA2D2Reader, f_colorType: CColorType, f_transform: transforms.Compose = None) -> None:
        self.m_dataPaths: list[tuple[str, str]] = list(zip(f_reader.m_framePaths, f_reader.m_labelPaths))
        self.m_transform: transforms.Compose = f_transform
        self.m_labelClasses: dict[str, str] = f_reader.m_labelClasses
        self.m_colorType: CColorType = f_colorType

    def __len__(self) -> int:
        return len(self.m_dataPaths)

    def __getitem__(self, f_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        f_framePath, f_labelPath = self.m_dataPaths[f_index]

        f_frame = Image.open(f_framePath).convert(self.m_colorType.ID)
        f_label = Image.open(f_labelPath).convert(self.m_colorType.ID)

        if self.m_transform:
            f_frame = self.m_transform(f_frame)
            f_label = self.m_transform(f_label)

        return f_frame, f_label
