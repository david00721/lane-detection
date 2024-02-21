from torchvision import transforms

from src.utils.parameters import CImageSize

m_transformation: transforms.Compose = transforms.Compose(
    [
        transforms.Resize((CImageSize.WIDTH, CImageSize.HEIGHT), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ]
)
