from src.reader.reader_base import CA2D2Reader


class CTrainingReader(CA2D2Reader):
    def __init__(self, f_datasetRoot: str, f_classesPath: str = r"C:\Git\AUDI_A2D2_dataset\class_list.json") -> None:
        super().__init__(f_datasetRoot, f_classesPath)


class CValidationReader(CA2D2Reader):
    def __init__(self, f_datasetRoot: str, f_classesPath: str = r"C:\Git\AUDI_A2D2_dataset\class_list.json") -> None:
        super().__init__(f_datasetRoot, f_classesPath)
