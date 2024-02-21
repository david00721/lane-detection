from dataclasses import dataclass, field
from torchvision import transforms as transform

# from dataclasses import field


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
class CTransformations:
    m_imageTransform = transform.Resize((CImageSize.WIDTH, CImageSize.HEIGHT), interpolation=transform.InterpolationMode.BILINEAR)
    m_labelTransform = transform.Resize((CImageSize.WIDTH, CImageSize.HEIGHT), interpolation=transform.InterpolationMode.NEAREST)


@dataclass
class CParameters:
    m_trainingDataPath: str = r"C:\Git\AUDI_A2D2_dataset\training"
    m_validationDataPath: str = r"C:\Git\AUDI_A2D2_dataset\validation"
    m_transformations: CTransformations = field(default_factory=CTransformations)
    m_imageSize: CImageSize = field(default_factory=CImageSize)
