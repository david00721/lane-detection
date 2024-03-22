import os
import ast
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.utils.parameters import CParameters
from src.reader.readers import CTrainingReader, CValidationReader
from src.dataset.datasets import CTrainingDataset, CValidationDataset


class CModelBase:
    def __init__(self, f_parameters: CParameters) -> None:
        self.m_trainingDataset: CTrainingDataset
        self.m_validationDataset: CValidationDataset
        self.m_trainingLoader: DataLoader
        self.m_validationLoader: DataLoader

        self.m_parameters: CParameters = f_parameters
        self.m_numberOfClasses: int
        self.m_model: torch.nn.Module
        self.m_trainingResults: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
        }

        self.m_amp = ast.literal_eval("True")
        self.m_lossFunction: torch.nn.Module
        self.m_optimizer: torch.optim.Optimizer
        self.m_scaler = torch.cuda.amp.GradScaler(enabled=self.m_amp)
        self.m_learningRate = f_parameters.m_trainingParameters.m_learningRate

    def loadSingleDataset(
        self,
        f_readerClass: CTrainingReader | CValidationReader,
        f_dataset: CTrainingDataset | CValidationDataset,
        f_datasetPath: str,
        f_maxNumberOfFrames: int = None,
    ) -> None:
        reader = f_readerClass(f_datasetPath, self.m_parameters.m_classListPath, f_maxNumberOfFrames)
        return f_dataset(reader, self.m_parameters.m_transformation)

    def loadDatasets(self) -> None:
        self.m_trainingDataset = self.loadSingleDataset(
            CTrainingReader, CTrainingDataset, self.m_parameters.m_trainingDataPath, self.m_parameters.m_trainingParameters.m_trainingFrameNumber
        )
        self.m_validationDataset = self.loadSingleDataset(
            CValidationReader, CValidationDataset, self.m_parameters.m_validationDataPath, self.m_parameters.m_trainingParameters.m_validationFrameNumber
        )

    def createDataLoaders(self) -> None:
        trainingParameters = self.m_parameters.m_trainingParameters
        self.m_trainingLoader = DataLoader(
            self.m_trainingDataset,
            batch_size=trainingParameters.m_batchSize,
            shuffle=trainingParameters.m_isShuffle,
            num_workers=trainingParameters.m_numberOfWorkers,
        )
        self.m_validationLoader = DataLoader(
            self.m_validationDataset,
            batch_size=trainingParameters.m_batchSize,
            shuffle=trainingParameters.m_isShuffle,
            num_workers=trainingParameters.m_numberOfWorkers,
        )

    def scheduleLearningRate(self, f_epoch: int) -> None:
        trainingParameters = self.m_parameters.m_trainingParameters

        if f_epoch <= trainingParameters.m_epochPeak:
            startLearningRate = self.m_learningRate * trainingParameters.m_learningRateWarmupRaito
            self.m_learningRate = startLearningRate + (f_epoch / trainingParameters.m_epochPeak) * (self.m_learningRate - startLearningRate)
        else:
            self.m_learningRate = self.m_learningRate * (trainingParameters.m_learningRateDecayPerEpoch) ** (f_epoch - trainingParameters.m_epochPeak)
        for p in self.m_optimizer.param_groups:
            p["lr"] = self.m_learningRate

        print(f"In epoch {f_epoch} learning rate: {self.m_learningRate}")

    def train(self) -> None:
        trainingParameters = self.m_parameters.m_trainingParameters
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.m_model.to(device)

        for epoch in range(0, trainingParameters.m_numberOfEpochs):
            self.scheduleLearningRate(epoch)
            correct, total = 0, 0

            for i, batch in enumerate(tqdm(self.m_trainingLoader)):
                self.m_model.train()
                inputs, masks = batch[0].to(device), batch[1].to(device).argmax(dim=3)
                self.m_optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=self.m_amp):
                    outputs = self.m_model(inputs)
                    training_loss = self.m_lossFunction(outputs, masks)
                    _, predicted = torch.max(outputs, 1)
                    total += masks.nelement()
                    correct += (predicted == masks).sum().item()

                self.m_scaler.scale(training_loss).backward()
                self.m_scaler.step(self.m_optimizer)
                self.m_scaler.update()

                if i > 0 and (i / float(trainingParameters.m_logFrequency)).is_integer():
                    train_accuracy = 100 * correct / total
                    val_losses = []
                    correct, total = 0, 0
                    self.m_model.eval()
                    with torch.no_grad():
                        for j, batch in enumerate(self.m_validationLoader):
                            inputs, masks = batch[0].to(device), batch[1].to(device).argmax(dim=3)
                            outputs = self.m_model(inputs)
                            val_loss = self.m_lossFunction(outputs, masks)
                            _, predicted = torch.max(outputs, 1)
                            total += masks.nelement()
                            correct += (predicted == masks).sum().item()
                            val_losses.append(val_loss)
                            if j * trainingParameters.m_batchSize >= trainingParameters.m_evaluationSize:  # evaluate on a subset of val set
                                break
                    self.savePredictions(inputs, masks, predicted, epoch, i)

                    avg_val_loss = torch.mean(torch.stack(val_losses))
                    val_accuracy = 100 * correct / total

                    self.updateTrainingResults(float(training_loss), float(avg_val_loss), train_accuracy, val_accuracy)

            print(
                f"Batch {i}:\nTraining_loss: {self.m_trainingResults['train_loss'][-1]:.4f}, \
                  Val_loss: {self.m_trainingResults['val_loss'][-1]:.4f}, \
                  Train_accuracy: {self.m_trainingResults['train_accuracy'][-1]:.2f}%, \
                  Val_accuracy: {self.m_trainingResults['val_accuracy'][-1]:.2f}%"
            )

            torch.save(self.m_model, os.path.join(trainingParameters.m_checkpointDirectory, f"model-epoch-{epoch}.pth"))

        torch.save(self.m_model, os.path.join(trainingParameters.m_checkpointDirectory, "final_model.pth"))
        df = pd.DataFrame(self.m_trainingResults)
        df.to_csv(os.path.join(trainingParameters.m_checkpointDirectory, "training_results.csv"))
        print("Finished Training")

    def savePredictions(self, f_inputs: torch.Tensor, f_masks: torch.Tensor, f_predicted: torch.Tensor, f_epoch: int, f_iter: int) -> None:
        plt.subplot(1, 3, 1)
        plt.imshow(np.transpose(f_inputs[0].squeeze().cpu().numpy(), (1, 2, 0)))
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(f_masks[0].squeeze().cpu().numpy())
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(f_predicted[0].squeeze().cpu().numpy())
        plt.axis("off")

        plt.savefig(f"{self.m_parameters.m_trainingParameters.m_predictionsDirectory}/prediction_epoch_{f_epoch}_iter_{f_iter}.png")

    def updateTrainingResults(self, f_loss: float, f_averageValLoss: float, f_trainingAccuracy: float, f_validationAccuracy: float) -> None:
        self.m_trainingResults["train_loss"].append(round(f_loss, 4))
        self.m_trainingResults["val_loss"].append(round(f_averageValLoss, 4))
        self.m_trainingResults["train_accuracy"].append(round(f_trainingAccuracy, 4))
        self.m_trainingResults["val_accuracy"].append(round(f_validationAccuracy, 4))
