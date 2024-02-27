import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from tqdm import tqdm
from torchvision.io import read_image, image
from torchvision import transforms as transform

from parameters import CParameters, FilteredLabels
from src.dataset.dataset_base import CDatasetBase


def preprocessFrames(
    f_originalPath: str, f_targetPath: str, f_transformation: transform.Compose, f_isLabel: bool, f_RGB2IDs: dict[tuple[int, int, int], int] = None
) -> None:
    l_cameraImages = sorted([img for img in os.listdir(f_originalPath) if os.path.splitext(img)[1].lower() in [".jpg", ".jpeg", ".png"]])
    for filename in tqdm(l_cameraImages):
        img = read_image(os.path.join(f_originalPath, filename), mode=image.ImageReadMode.RGB)
        img = f_transformation(img)

        if f_isLabel:
            id_map = CDatasetBase.convertLabelToClassIDMap(img, f_RGB2IDs)
            torch.save(id_map.type(torch.int64), os.path.join(f_targetPath, filename))
            continue

        torch.save(torch.div(img, 255), os.path.join(f_targetPath, filename))


def main() -> None:
    parameters = CParameters()

    with open(parameters.m_classListPath, "r") as json_file:
        all_classes = json.load(json_file)

    RGB2IDs = CDatasetBase.convertClassesToIDs({k: v for k, v in all_classes.items() if v in FilteredLabels})

    preprocessedDataPath = os.path.join(parameters.m_datasetRoot, "preprocessed")
    originalDataPath = os.path.join(parameters.m_datasetRoot, "training")

    os.makedirs(preprocessedDataPath, exist_ok=True)

    l_sceneFolders = [d for d in os.listdir(originalDataPath) if os.path.isdir(os.path.join(originalDataPath, d))]

    for idx, sceneFolder in enumerate(l_sceneFolders):
        print(f"Iteration: {idx}\nPreprocessing {sceneFolder}")
        os.makedirs(os.path.join(preprocessedDataPath, sceneFolder), exist_ok=True)

        camera = os.path.join(preprocessedDataPath, sceneFolder, "camera", "cam_front_center")
        label = os.path.join(preprocessedDataPath, sceneFolder, "label", "cam_front_center")

        os.makedirs(camera, exist_ok=True)
        os.makedirs(label, exist_ok=True)

        preprocessFrames(os.path.join(originalDataPath, sceneFolder, "camera", "cam_front_center"), camera, parameters.m_transformation, False)
        preprocessFrames(os.path.join(originalDataPath, sceneFolder, "label", "cam_front_center"), label, parameters.m_transformation, True, RGB2IDs)


if __name__ == "__main__":
    main()
