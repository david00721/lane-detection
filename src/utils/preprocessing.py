import os
import sys
from tqdm import tqdm
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from parameters import CParameters, CImageParameters


def preprocessFrames(f_originalPath: str, f_targetPath: str) -> None:
    l_cameraImages = sorted([img for img in os.listdir(f_originalPath) if os.path.splitext(img)[1].lower() in [".jpg", ".jpeg", ".png"]])
    for filename in tqdm(l_cameraImages):
        img = Image.open(os.path.join(f_originalPath, filename)).resize((CImageParameters.WIDTH, CImageParameters.HEIGHT), Image.NEAREST)
        img.save(os.path.join(f_targetPath, filename))


def main() -> None:
    parameters = CParameters()

    preprocessedDataPath = os.path.join(parameters.m_datasetRoot, "preprocessed")
    originalDataPath = os.path.join(parameters.m_datasetRoot, "training")

    os.makedirs(preprocessedDataPath, exist_ok=True)

    l_sceneFolders = [d for d in os.listdir(originalDataPath) if os.path.isdir(os.path.join(originalDataPath, d))]

    for idx, sceneFolder in enumerate(l_sceneFolders):
        print(f"Iteration: {idx}\nPreprocessing {sceneFolder}")
        os.makedirs(os.path.join(preprocessedDataPath, sceneFolder), exist_ok=True)

        for cam in ["cam_front_center", "cam_front_right", "cam_front_left", "cam_rear_center"]:
            if not os.path.exists(os.path.join(originalDataPath, sceneFolder, "camera", cam)):
                continue

            camera = os.path.join(preprocessedDataPath, sceneFolder, "camera", cam)
            label = os.path.join(preprocessedDataPath, sceneFolder, "label", cam)

            os.makedirs(camera, exist_ok=True)
            os.makedirs(label, exist_ok=True)

            preprocessFrames(os.path.join(originalDataPath, sceneFolder, "camera", cam), camera)
            preprocessFrames(os.path.join(originalDataPath, sceneFolder, "label", cam), label)


if __name__ == "__main__":
    main()
