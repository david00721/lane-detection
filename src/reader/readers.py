from src.reader.reader_base import CA2D2Reader


class CTrainingReader(CA2D2Reader):
    def __init__(self, f_datasetRoot: str, f_classesPath: str = r"C:\Git\AUDI_A2D2_dataset\class_list.json", f_maxNumberOfFrames: int = None) -> None:
        super().__init__(f_datasetRoot, f_classesPath, f_maxNumberOfFrames)


class CValidationReader(CA2D2Reader):
    def __init__(self, f_datasetRoot: str, f_classesPath: str = r"C:\Git\AUDI_A2D2_dataset\class_list.json", f_maxNumberOfFrames: int = None) -> None:
        super().__init__(f_datasetRoot, f_classesPath, f_maxNumberOfFrames)
