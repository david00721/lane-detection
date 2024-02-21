from dataclasses import dataclass

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
    WIDTH: int = 640
    HEIGHT: int = 640


# @dataclass
# class CImageParameters:
#     IMAGE_SIZE: CImageSize = field(default_factory=CImageSize)
#     COLOR_TYPE: CRGBType = field(default_factory=CRGBType)


# @dataclass
# class CParameters:
#     m_imageParameters: CImageParameters = field(default_factory=CImageParameters)
