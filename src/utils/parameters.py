import os
from enum import Enum
from dataclasses import dataclass, field
import torch


@dataclass(frozen=True)
class CFilteredA2D2Classnames(Enum):
    """Enum class representing the filtered segmentation classes from the A2D2 dataset."""

    BACKGROUND = "Background"
    SOLID_LINE = "Solid line"
    ZEBRA_CROSSING = "Zebra crossing"
    RD_RESTRICTED_AREA = "RD restricted area"
    DRIVABLE_COBBLESTONE = "Drivable cobblestone"
    TRAFFIC_GUIDE_OBJ = "Traffic guide obj."
    DASHED_LINE = "Dashed line"
    RD_NORMAL_STREET = "RD normal street"


@dataclass(frozen=True)
class CSegmentationClass:
    """Dataclass representing the RGB to ID conversion."""

    ID: int
    CLASS_NAME: str
    HEXA: str
    RGB: tuple[int, int, int]


@dataclass(frozen=True)
class CImageParameters:
    """Dataclass representing the image extentions and size for the model."""

    WIDTH: int = 416
    HEIGHT: int = 224

    @dataclass(frozen=True)
    class CImageExtentions(Enum):
        """Enum class representing the image extentions."""

        JPG = ".jpg"
        JPEG = ".jpeg"
        PNG = ".png"


@dataclass(frozen=True)
class CDatasetPaths:
    """Dataclass representing the dataset paths."""

    ROOT: str
    TRAINING: str = field(default_factory=str, init=False)
    CLASS_LIST: str = field(default_factory=str, init=False)

    def __post_init__(self):
        """Post initialization method to set the training and class list paths."""
        object.__setattr__(self, "TRAINING", os.path.join(self.ROOT, "preprocessed"))
        object.__setattr__(self, "CLASS_LIST", os.path.join(self.ROOT, "class_list.json"))


@dataclass(frozen=True)
class CRepositoryPaths:
    """Dataclass representing the repository paths."""

    ROOT: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    CHECKPOINTS: str = os.path.join(ROOT, "checkpoints")
    PREDICTIONS: str = os.path.join(ROOT, "predictions")
    ENCODED_TENSORS: str = os.path.join(ROOT, "encoded_tensors")

    """Post initialization method to create the directories if they do not exist."""
    os.makedirs(CHECKPOINTS, exist_ok=True)
    os.makedirs(PREDICTIONS, exist_ok=True)
    os.makedirs(ENCODED_TENSORS, exist_ok=True)


@dataclass(frozen=True)
class CModelParameters:
    """Dataclass representing the model parameters."""

    ENCODER_NAME: str = "resnet34"
    IN_CHANNELS: int = 3
    ENCODER_DEPTH: int = 3
    ENCODER_WEIGHTS: str = "imagenet"
    DROP_OUT: float = 0.35


@dataclass(frozen=True)
class CTrainingResults:
    """Dataclass representing the training results."""

    TRAINING_LOSS: float
    TRAINING_ACCURACY: float
    VALIDATION_LOSS: float
    VALIDATION_ACCURACY: float


@dataclass(frozen=True)
class CTrainingParametersBase:
    """Dataclass representing the training parameters."""

    BATCH_SIZE: int
    NUMBER_OF_EPOCHS: int
    ENABLE_SHUFFLE: bool
    CHUNK_SIZE: int
    LEARNING_RATE: float
    LOG_FREQUENCY: int
    TRAINING_DATASET_PERCENTAGE: float
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class CTrainingParametersLocal(CTrainingParametersBase):
    """Dataclass representing the training parameters for local training."""

    BATCH_SIZE: int = 32
    NUMBER_OF_EPOCHS: int = 1
    ENABLE_SHUFFLE: bool = True
    CHUNK_SIZE: int = 100
    LEARNING_RATE: float = 0.01
    LOG_FREQUENCY: int = 5
    TRAINING_DATASET_PERCENTAGE: float = 0.8


@dataclass(frozen=True)
class CEncodingParameters:
    """Dataclass representing the encoding parameters."""

    BATCH_SIZE: int = 64
    SHUFFLE: bool = True


@dataclass(frozen=True)
class CParameters:
    """Dataclass representing the parameters for the training script."""

    DATASET_PATHS: CDatasetPaths
    TRAINING_PARAMETERS: CTrainingParametersBase = field(default_factory=CTrainingParametersBase)


def create_parameters(f_datasetPath: str) -> CParameters:
    """
    Create parameters for training script.

    Args:
        f_datasetPath (str): The path to the dataset.

    Returns:
        CParameters: The created parameters object.
    """
    parameters = CParameters(DATASET_PATHS=CDatasetPaths(ROOT=f_datasetPath), TRAINING_PARAMETERS=CTrainingParametersLocal())
    return parameters
