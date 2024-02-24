from dataclasses import dataclass, field
from torchvision import transforms as transform


FilteredLabels: list = [
    "Background",
    "Solid line",
    # "Non-drivable street",
    "Zebra crossing",
    "RD restricted area",
    "Drivable cobblestone",
    # "Slow drive area",
    # "Parking area",
    # "Painted driv. instr.",
    "Traffic guide obj.",
    "Dashed line",
    "RD normal street",
]


@dataclass
class CColorType:
    ID: str
    CHANNELS: int


@dataclass
class CRGBType(CColorType):
    ID: str = "RGB"
    CHANNELS: int = 3


@dataclass
class CGrayScaleType(CColorType):
    ID: str = "L"
    CHANNELS: int = 1


@dataclass
class CImageSize:
    WIDTH: int = 450
    HEIGHT: int = 450


@dataclass
class CTrainingParameters:
    m_batchSize: int = 16
    m_isShuffle: bool = True
    m_numberOfWorkers: int = 4
    m_numberOfEpochs: int = 10
    m_useWeights: bool = False
    m_epochPeak: int = 2
    m_learningRate: float = 0.01
    m_learningRateWarmupRaito: float = 1.0
    m_learningRateDecayPerEpoch: float = 1.0
    m_logFrequency: int = 1
    m_evaluationSize: int = 30
    m_checkpointDirectory: str = r"C:\Git\lane-detection\src\checkpoints"
    m_momentum: float = 0.95


@dataclass
class CParameters:
    m_isLocalTraining: bool = True
    m_trainingDataPath: str = r"C:\Git\AUDI_A2D2_dataset\training"
    m_validationDataPath: str = r"C:\Git\AUDI_A2D2_dataset\validation"
    m_classListPath: str = r"C:\Git\AUDI_A2D2_dataset\class_list.json"
    m_transformation = transform.Compose([transform.Resize((CImageSize.WIDTH, CImageSize.HEIGHT))])
    m_imageSize: CImageSize = field(default_factory=CImageSize)
    m_trainingParameters: CTrainingParameters = field(default_factory=CTrainingParameters)
    m_numberOfClasses: int = len(FilteredLabels)
