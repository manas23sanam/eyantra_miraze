import abc
from typing import Optional, Any
import numpy as np

class BaseExercise(abc.ABC):

    def __init__(self, config: dict, intrinsics: Any):
        self.config = config
        self.intrinsics = intrinsics

    @abc.abstractmethod
    def process_frame(self, color_image: np.ndarray, depth_frame: Any, depth_image: np.ndarray) -> dict:
        pass
