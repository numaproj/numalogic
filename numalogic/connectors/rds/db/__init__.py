import importlib
import os

# value is an array of classes
CLASS_TYPE = None

for file_name in os.listdir(os.path.dirname(__file__)):
    if file_name.endswith("Fetcher.py"):
        class_name = file_name[:-3]
        module = importlib.import_module(f'.{class_name}', __name__)
        CLASS_TYPE = getattr(module, class_name)


