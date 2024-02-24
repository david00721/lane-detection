import os
import json


class CA2D2Reader:
    def __init__(self, f_datasetRoot: str, f_classesPath: str, f_maxNumberOfFrames: int = None) -> None:
        self.m_datasetRoot: str = f_datasetRoot
        self.m_framePaths: list[str] = []
        self.m_labelPaths: list[str] = []
        self.m_imageExtensions: list[str] = [".jpg", ".jpeg", ".png"]
        self.m_labelClasses: dict[str, str] = {}
        self.m_classesPath: str = f_classesPath

        self.collectA2D2DataPaths()
        self.loadClasses()

        if f_maxNumberOfFrames:
            self.m_framePaths = self.m_framePaths[:f_maxNumberOfFrames]
            self.m_labelPaths = self.m_labelPaths[:f_maxNumberOfFrames]

    def collectA2D2DataPaths(self) -> None:
        """Collect image paths and label paths from the A2D2 dataset."""
        l_sceneFolders: list[str] = [d for d in os.listdir(self.m_datasetRoot) if os.path.isdir(os.path.join(self.m_datasetRoot, d))]

        for sceneFolder in l_sceneFolders:
            frames_folders = [f.path for f in os.scandir(os.path.join(self.m_datasetRoot, sceneFolder, "camera")) if f.is_dir()]
            labels_folders = [f.path for f in os.scandir(os.path.join(self.m_datasetRoot, sceneFolder, "label")) if f.is_dir()]

            self.collectFramePaths(frames_folders)
            self.collectLabelPaths(labels_folders)

    def collectFramePaths(self, f_folders: list[str]) -> None:
        for path in f_folders:
            l_cameraImages = sorted([img for img in os.listdir(path) if os.path.splitext(img)[1].lower() in self.m_imageExtensions])
            self.m_framePaths.extend([os.path.join(path, camera_img) for camera_img in l_cameraImages])

    def collectLabelPaths(self, f_folders: list[str]) -> None:
        for path in f_folders:
            l_labelImages = sorted([img for img in os.listdir(path) if os.path.splitext(img)[1].lower() in self.m_imageExtensions])
            self.m_labelPaths.extend([os.path.join(path, label_img) for label_img in l_labelImages])

    def loadClasses(self) -> None:
        with open(self.m_classesPath, "r") as json_file:
            self.m_labelClasses = json.load(json_file)
