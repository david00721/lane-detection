import os
import json


class CA2D2Reader:
    def __init__(self, f_datasetRoot: str, f_classesPath: str = r"C:\Git\AUDI_A2D2_dataset\class_list.json") -> None:
        self.m_datasetRoot: str = f_datasetRoot
        self.m_framePaths: list[str] = []
        self.m_labelPaths: list[str] = []
        self.m_imageExtensions: list[str] = [".jpg", ".jpeg", ".png"]
        self.m_labelClasses: dict[str, str] = {}
        self.m_classesPath: str = f_classesPath

        self.collectA2D2DataPaths()
        self.loadClasses()

    def collectA2D2DataPaths(self) -> None:
        """Collect image paths and label paths from the A2D2 dataset."""
        l_sceneFolders: list[str] = [d for d in os.listdir(self.m_datasetRoot) if os.path.isdir(os.path.join(self.m_datasetRoot, d))]

        for sceneFolder in l_sceneFolders:
            subfolders = [f.path for f in os.scandir(os.path.join(self.m_datasetRoot, sceneFolder, "camera")) if f.is_dir()]

            for path in subfolders:
                l_cameraImages = sorted([img for img in os.listdir(path) if os.path.splitext(img)[1].lower() in self.m_imageExtensions])
                l_labelImages = sorted([img for img in os.listdir(path) if os.path.splitext(img)[1].lower() in self.m_imageExtensions])

                self.m_framePaths.extend([os.path.join(path, camera_img) for camera_img in l_cameraImages])
                self.m_labelPaths.extend([os.path.join(path, label_img) for label_img in l_labelImages])

    def loadClasses(self) -> None:
        with open(self.m_classesPath, "r") as json_file:
            self.m_labelClasses = json.load(json_file)
