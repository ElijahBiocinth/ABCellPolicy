from abc import ABC, abstractmethod
from typing import List
import numpy as np

class BaseDetector(ABC):
    @abstractmethod
    def __init__(self, model_path: str, device: str): ...
    @abstractmethod
    def detect(self, tile_img: np.ndarray) -> List[np.ndarray]: ...
