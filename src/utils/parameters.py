import os
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
class CImageSize:
    WIDTH: int = 416
    HEIGHT: int = 224


@dataclass
class CTrainingParameters:
    m_batchSize: int = 32
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
    m_checkpointDirectory: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "checkpoints")
    m_predictionsDirectory: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "predictions")
    m_momentum: float = 0.95
    m_trainingFrameNumber: int = 1000
    m_validationFrameNumber: int = 50


@dataclass
class CParameters:
    m_datasetRoot: str = "C:/Git/AUDI_A2D2_dataset"
    m_trainingDataPath: str = os.path.join(m_datasetRoot, "training")
    m_validationDataPath: str = os.path.join(m_datasetRoot, "validation")
    m_classListPath: str = os.path.join(m_datasetRoot, "class_list.json")
    m_transformation = transform.Compose([transform.Resize((CImageSize.HEIGHT, CImageSize.WIDTH), antialias=True)])
    m_imageSize: CImageSize = field(default_factory=CImageSize)
    m_trainingParameters: CTrainingParameters = field(default_factory=CTrainingParameters)
    m_numberOfClasses: int = len(FilteredLabels)
