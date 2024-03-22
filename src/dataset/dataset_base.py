import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image, image

from src.utils.parameters import CImageSize, FilteredLabels
from src.reader.reader_base import CA2D2Reader


class CDatasetBase(Dataset):

    def __init__(self, f_reader: CA2D2Reader, f_transformation: transforms.Compose) -> None:
        self.m_dataPaths: list[tuple[str, str]] = list(zip(f_reader.m_framePaths, f_reader.m_labelPaths))
        self.m_transformation: transforms.Compose = f_transformation
        self.m_filteredClasses: dict[str, int] = {k: v for k, v in f_reader.m_labelClasses.items() if v in FilteredLabels}
        self.m_RGB2IDs: dict[tuple[int, int, int], int] = self.convertClassesToIDs(self.m_filteredClasses)
        self.m_numberOfClasses: int = len(self.m_filteredClasses)

    def __len__(self) -> int:
        return len(self.m_dataPaths)

    def __getitem__(self, f_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        framePath, labelPath = self.m_dataPaths[f_index]

        frame = read_image(framePath, mode=image.ImageReadMode.RGB)
        frame = self.m_transformation(frame)

        label = read_image(labelPath, mode=image.ImageReadMode.RGB)
        label = self.m_transformation(label)

        id_map = CDatasetBase.convertLabelToClassIDMap(label, self.m_RGB2IDs)

        return torch.div(frame, 255), torch.nn.functional.one_hot(id_map.long())

    @staticmethod
    def convertHexaClassColorsToRBG(f_labelClasses: dict[int, int]) -> dict[str, tuple[int, int, int]]:
        return {k: tuple(int(k.lstrip("#")[i : i + 2], 16) for i in range(0, 6, 2)) for k in f_labelClasses.keys()}

    @staticmethod
    def convertClassesToIDs(f_labelClasses: dict[int, int]) -> dict[tuple[int, int, int], int]:
        hexaToTGB = CDatasetBase.convertHexaClassColorsToRBG(f_labelClasses)
        return {r: i for i, r in enumerate(hexaToTGB.values())}

    @staticmethod
    def convertLabelToClassIDMap(f_label: torch.Tensor, f_RGB2IDs: dict[tuple[int, int, int], int]) -> torch.Tensor:
        mask = torch.zeros(CImageSize.HEIGHT, CImageSize.WIDTH)
        for rgb_code, class_id in f_RGB2IDs.items():
            color_mask = f_label == torch.Tensor(rgb_code).reshape([3, 1, 1])
            seg_mask = color_mask.sum(dim=0) == 3
            mask[seg_mask] = class_id

        return mask
