import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image, image

from data.A2D2_reader import CA2D2Reader
from utils.parameters import CSegmentationClass
from utils.utils import convert_label_to_class_id_map


class CA2D2Dataset(Dataset):
    """A2D2 Dataset Class"""

    def __init__(self, f_reader: CA2D2Reader, f_transformationImage: transforms.Compose = None, f_transformationLabel: transforms.Compose = None) -> None:
        self.m_dataPaths: list[tuple[str, str]] = list(zip(f_reader.m_framePaths, f_reader.m_labelPaths))
        self.m_transformationImage: transforms.Compose = f_transformationImage
        self.m_transformationLabel: transforms.Compose = f_transformationLabel
        self.m_labelClasses: list[CSegmentationClass] = f_reader.m_labelClasses

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: The number of data paths in the dataset.
        """
        return len(self.m_dataPaths)

    def __getitem__(self, f_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the item at the given index from the dataset. Labels are One-Hot encoded.

        Args:
            f_index (int): The index of the item to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the frame and label tensors.
        """
        framePath, labelPath = self.m_dataPaths[f_index]

        frame = read_image(framePath, mode=image.ImageReadMode.RGB)
        if self.m_transformationImage:
            frame = self.m_transformationImage(frame)

        label = read_image(labelPath, mode=image.ImageReadMode.RGB)
        if self.m_transformationLabel:
            label = self.m_transformationLabel(label)

        id_map = convert_label_to_class_id_map(label, self.m_labelClasses)

        # return (
        #     torch.div(frame, 255).to(torch.float16),
        #     torch.nn.functional.one_hot(id_map.long(), num_classes=len(self.m_labelClasses)).permute(2, 0, 1).to(torch.int8),
        # )
        return torch.div(frame, 255).to(torch.float16), id_map.long().to(torch.int8)
