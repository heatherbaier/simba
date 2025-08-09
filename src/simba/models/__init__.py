# src/simba/models/__init__.py
from ..core.registries import ModelRegistry

# Import concrete model classes
from .resnet18 import ResNet18Classifier
from .tang2015 import Tang2015Classifier  # or Tang2015Wrapper if that's your class name

# Register them under CLI-friendly names
ModelRegistry.register("resnet18")(ResNet18Classifier)
ModelRegistry.register("tang2015")(Tang2015Classifier)
