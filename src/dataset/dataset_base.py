import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image, image

from src.utils.parameters import CColorType, CImageSize
from src.reader.reader_base import CA2D2Reader


class CDatasetBase(Dataset):

    def __init__(self, f_reader: CA2D2Reader, f_colorType: CColorType, f_transform: transforms.Resize, f_targetTransform: transforms.Resize) -> None:
        self.m_dataPaths: list[tuple[str, str]] = list(zip(f_reader.m_framePaths, f_reader.m_labelPaths))
        self.m_transform: transforms.Resize = f_transform
        self.m_targetTransform: transforms.Resize = f_targetTransform
        self.m_labelClasses: dict[str, str] = f_reader.m_labelClasses
        self.m_colorType: CColorType = f_colorType
        self.hex2rgb = {k: tuple(int(k.strip("#")[i : i + 2], 16) for i in (0, 2, 4)) for k in self.m_labelClasses.keys()}
        self.m_RGB2IDs = {r: i for i, r in enumerate(self.hex2rgb.values())}

    def __len__(self) -> int:
        return len(self.m_dataPaths)

    def __getitem__(self, f_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        framePath, labelPath = self.m_dataPaths[f_index]

        frame = read_image(framePath, mode=image.ImageReadMode.RGB)
        frame = self.m_transform(frame)

        label = read_image(labelPath, mode=image.ImageReadMode.RGB)
        label = self.m_targetTransform(label)

        mask = torch.zeros(CImageSize.WIDTH, CImageSize.HEIGHT)
        for rgb, cid in self.m_RGB2IDs.items():
            color_mask = label == torch.Tensor(rgb).reshape([3, 1, 1])
            seg_mask = color_mask.sum(dim=0) == 3
            mask[seg_mask] = cid

        return torch.div(frame, 255), mask.type(torch.int64)
