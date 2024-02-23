import os
import ast
import torch
import time
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.utils.parameters import CParameters
from src.reader.readers import CTrainingReader, CValidationReader
from src.dataset.datasets import CTrainingDataset, CValidationDataset


class CSegmentationModel:

    def __init__(self, f_parameters: CParameters) -> None:
        self.m_trainingDataset: CTrainingDataset
        self.m_validationDataset: CValidationDataset
        self.m_trainingLoader: DataLoader
        self.m_validationLoader: DataLoader

        self.m_parameters: CParameters = f_parameters
        self.m_numberOfClasses: int = 55  # self.m_trainingDataset.m_numberOfClasses  # TODO: make this automatic

        self.m_model = torch.hub.load(
            "pytorch/vision:v0.9.1", "deeplabv3_mobilenet_v3_large", weights=f_parameters.m_trainingParameters.m_useWeights, num_classes=self.m_numberOfClasses
        )

        self.m_amp = ast.literal_eval("True")
        self.m_lossFunction = torch.nn.CrossEntropyLoss()
        self.m_optimizer = torch.optim.SGD(
            self.m_model.parameters(), lr=f_parameters.m_trainingParameters.m_learningRate, momentum=f_parameters.m_trainingParameters.m_momentum
        )
        self.m_scaler = torch.cuda.amp.GradScaler(enabled=self.m_amp)
        self.m_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.m_learningRate = f_parameters.m_trainingParameters.m_learningRate

        self.m_model.to(self.m_device)

    def loadSingleDataset(self, f_readerClass, f_dataset, f_datasetPath: str) -> None:
        reader = f_readerClass(f_datasetPath)
        return f_dataset(reader, self.m_parameters.m_transformations.m_imageTransform, self.m_parameters.m_transformations.m_labelTransform)

    def loadDatasets(self) -> None:
        self.m_trainingDataset = self.loadSingleDataset(CTrainingReader, CTrainingDataset, self.m_parameters.m_trainingDataPath)
        self.m_validationDataset = self.loadSingleDataset(CValidationReader, CValidationDataset, self.m_parameters.m_validationDataPath)

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

    def processBatches(self, f_epoch: int) -> None:
        start_time = time.time()
        correct, total = 0, 0
        for i, batch in enumerate(tqdm(self.m_trainingLoader)):
            self.m_model.train()
            inputs = batch[0].to(self.m_device)
            masks = batch[1].to(self.m_device)
            self.m_optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.m_amp):
                outputs = self.m_model(inputs)
                loss = self.m_lossFunction(outputs["out"], masks.long())
                _, predicted = torch.max(outputs["out"], 1)
                total += masks.nelement()
                correct += (predicted == masks.long()).sum().item()

            self.m_scaler.scale(loss).backward()
            self.m_scaler.step(self.m_optimizer)
            self.m_scaler.update()

            if i > 0 and (i / float(self.m_parameters.m_trainingParameters.m_logFrequency)).is_integer():
                train_accuracy = 100 * correct / total
                stop_time = time.time()
                val_losses = []
                correct = 0
                total = 0
                self.m_model.eval()
                with torch.no_grad():
                    # validation dataloader takes 30s to load first batch :(...
                    for j, batch in enumerate(tqdm(self.m_validationLoader)):
                        inputs = batch[0].to(self.m_device)
                        masks = batch[1].to(self.m_device)
                        outputs = self.m_model(inputs)
                        val_loss = self.m_lossFunction(outputs["out"], masks)
                        _, predicted = torch.max(outputs["out"], 1)
                        total += masks.nelement()
                        correct += (predicted == masks).sum().item()
                        val_losses.append(val_loss)
                        if (
                            j * self.m_parameters.m_trainingParameters.m_batchSize >= self.m_parameters.m_trainingParameters.m_evaluationSize
                        ):  # evaluate on a subset of val set
                            break
                avg_val_loss = torch.mean(torch.stack(val_losses))
                val_accuracy = 100 * correct / total

                # print metrics
                # throughput = float((i + 1) * self.m_parameters.m_trainingParameters.m_batchSize) / (stop_time - start_time)
                print("processed {} records in {}s".format(i * self.m_parameters.m_trainingParameters.m_batchSize, stop_time - start_time))
                print(
                    f"batch {i}: Training_loss: {loss:.4f}, Val_loss: {avg_val_loss:.4f}, Train_accuracy: {train_accuracy:.2f}%, Val_accuracy: {val_accuracy:.2f}%"
                )

                # save model twice ("latest" and versioned)
                checkpoint_name = "model-epoch{}-iter{}.pth".format(f_epoch, i)
                torch.save(self.m_model, os.path.join(self.m_parameters.m_trainingParameters.m_checkpointDirectory, checkpoint_name))
                torch.save(self.m_model, os.path.join(self.m_parameters.m_trainingParameters.m_checkpointDirectory, "latest_model.pth"))

    def train(self) -> None:
        for epoch in range(0, self.m_parameters.m_trainingParameters.m_numberOfEpochs):
            self.scheduleLearningRate(epoch)
            self.processBatches(epoch)

            # we save the final model in the checkpoint location, for consistency
            torch.save(self.m_model, os.path.join(self.m_parameters.m_trainingParameters.m_checkpointDirectory, "final_model.pth"))
