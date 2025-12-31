# C.A. Dupin - Core modules
from .image_matcher import ImageMatcher
from .model_trainer import ModelTrainer, SiameseNetwork
from .human_feedback import HumanFeedbackLoop
from .roi_manager import ROIManager
from .camera_manager import CameraManager
from .visual_interface import VisualInterface
from .language_manager import LanguageManager
from .modules import ModuleManager

__all__ = [
    'ImageMatcher',
    'ModelTrainer', 
    'SiameseNetwork',
    'HumanFeedbackLoop',
    'ROIManager',
    'CameraManager', 
    'VisualInterface',
    'LanguageManager',
    'ModuleManager'
]