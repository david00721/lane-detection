import os
import json

from utils.parameters import CImageParameters, CFilteredA2D2Classnames
from utils.utils import create_segmentation_classlist


class CA2D2Reader:
    def __init__(self, f_datasetRoot: str, f_classesPath: str) -> None:
        self.m_datasetRoot: str = f_datasetRoot
        self.m_framePaths: list[str] = []
        self.m_labelPaths: list[str] = []
        self.m_labelClasses: dict[str, str] = {}
        self.m_classesPath: str = f_classesPath

        self.collectA2D2DataPaths()
        self.loadClasses()

    def collectA2D2DataPaths(self) -> None:
        """
        Collects the paths of A2D2 dataset frames and labels.

        Returns:
            None
        """
        l_sceneFolders: list[str] = [d for d in os.listdir(self.m_datasetRoot) if os.path.isdir(os.path.join(self.m_datasetRoot, d))]

        for sceneFolder in l_sceneFolders:
            frames_folders = [f.path for f in os.scandir(os.path.join(self.m_datasetRoot, sceneFolder, "camera")) if f.is_dir()]
            labels_folders = [f.path for f in os.scandir(os.path.join(self.m_datasetRoot, sceneFolder, "label")) if f.is_dir()]

            self.collectPaths(frames_folders, self.m_framePaths)
            self.collectPaths(labels_folders, self.m_labelPaths)

    @classmethod
    def collectPaths(cls, f_folders: list[str], f_dataPaths: list[str]) -> None:
        """
        Collects the paths of images in the given folders and appends them to the provided list.

        Args:
            f_folders (list[str]): A list of folder paths.
            f_dataPaths (list[str]): A list to store the collected image paths.

        Returns:
            None
        """
        for path in f_folders:
            new_paths = sorted([img for img in os.listdir(path) if os.path.splitext(img)[1].lower() in [e.value for e in CImageParameters.CImageExtentions]])
            f_dataPaths.extend([os.path.join(path, img) for img in new_paths])

    def loadClasses(self) -> None:
        """
        Loads the label classes from a JSON file and creates a segmentation classlist based on the filtered classlist.
        Classlist is converted to list of SegmentationClass objects.

        Returns:
            None
        """
        with open(self.m_classesPath, "r") as json_file:
            labelClasses = json.load(json_file)

        self.m_labelClasses = create_segmentation_classlist({k: v for k, v in labelClasses.items() if v in [e.value for e in CFilteredA2D2Classnames]})
