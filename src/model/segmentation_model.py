import torch

from src.utils.parameters import CParameters
from src.model.model_base import CModelBase


class CSegmentationModel(CModelBase):
    def __init__(self, f_parameters: CParameters) -> None:
        super().__init__(f_parameters)
        self.m_numberOfClasses: int = f_parameters.m_numberOfClasses

        self.m_model = torch.hub.load(
            "pytorch/vision:v0.9.1", "deeplabv3_mobilenet_v3_large", weights=f_parameters.m_trainingParameters.m_useWeights, num_classes=self.m_numberOfClasses
        )

        self.m_lossFunction = torch.nn.CrossEntropyLoss()
        self.m_optimizer = torch.optim.SGD(
            self.m_model.parameters(), lr=f_parameters.m_trainingParameters.m_learningRate, momentum=f_parameters.m_trainingParameters.m_momentum
        )
