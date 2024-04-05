import importlib
import os

# value is an array of classes
CLASS_TYPE = None

"""This code snippet is a loop that iterates over the files in the directory of the current file. It checks if each 
file ends with "Fetcher.py" and if so, extracts the class name by removing the ".py" extension. It then imports the 
module corresponding to the class and retrieves the class object using the class name. Finally, it assigns the class 
object to the CLASS_TYPE variable. This code snippet is used to dynamically import and assign class objects based on 
file names."""

for file_name in os.listdir(os.path.dirname(__file__)):
    if file_name.endswith("Fetcher.py"):
        class_name = file_name[:-3]
        module = importlib.import_module(f'.{class_name}', __name__)
        CLASS_TYPE = getattr(module, class_name)
