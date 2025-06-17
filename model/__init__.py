import os
import importlib
from pathlib import Path
from model.base_model import BaseForecastModel

def discover_model_classes():
    model_dir = Path(__file__).parent
    model_classes = {}

    for file in os.listdir(model_dir):
        # Skip non-python files, __init__.py, and base_model.py
        if not file.endswith('.py') or file in ('__init__.py', 'base_model.py'):
            continue

        module_name = f"model.{file[:-3]}"  # Strip .py extension
        module = importlib.import_module(module_name)

        # Cari class turunan BaseForecastModel yang ada di modul
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            try:
                if (
                    isinstance(attr, type) 
                    and issubclass(attr, BaseForecastModel) 
                    and attr is not BaseForecastModel
                ):
                    model_classes[attr_name] = attr
            except Exception:
                continue

    return model_classes

ALL_MODEL_CLASSES = discover_model_classes()