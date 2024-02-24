from src.dataset.dataset_base import CDatasetBase
from torchvision import transforms

from src.reader.reader_base import CA2D2Reader


class CTrainingDataset(CDatasetBase):
    def __init__(self, f_reader: CA2D2Reader, f_transformation: transforms.Compose) -> None:
        super().__init__(f_reader, f_transformation)


class CValidationDataset(CDatasetBase):
    def __init__(self, f_reader: CA2D2Reader, f_transformation: transforms.Compose) -> None:
        super().__init__(f_reader, f_transformation)
