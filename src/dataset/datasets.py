from src.dataset.dataset_base import CDatasetBase
from torchvision import transforms

from src.reader.reader_base import CA2D2Reader
from src.utils.parameters import CColorType


class CTrainingDataset(CDatasetBase):
    def __init__(self, f_reader: CA2D2Reader, f_colorType: CColorType, f_transform: transforms.Resize, f_targetTransform: transforms.Resize) -> None:
        super().__init__(f_reader, f_colorType, f_transform, f_targetTransform)


class CValidationDataset(CDatasetBase):
    def __init__(self, f_reader: CA2D2Reader, f_colorType: CColorType, f_transform: transforms.Resize, f_targetTransform: transforms.Resize) -> None:
        super().__init__(f_reader, f_colorType, f_transform, f_targetTransform)
