import os
from contextlib import contextmanager
from importlib.util import spec_from_file_location, module_from_spec
from types import ModuleType

from setuptools import setup


def _load_py_module(name: str, location: str) -> ModuleType:
    spec = spec_from_file_location(name, location)
    assert spec, f"Failed to load module {name} from {location}"
    py = module_from_spec(spec)
    assert spec.loader, f"ModuleSpec.loader is None for {name} from {location}"
    spec.loader.exec_module(py)
    return py


@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


pkg_name = os.getenv("PKG_NAME", "numalogic")
if pkg_name == "numalogic":
    with cd("src/core"):
        setup()

elif pkg_name == "connectors":
    setup_module = _load_py_module(
        name="connectors_setup", location="src/test_connectors/__setup__.py"
    )
elif pkg_name == "registry":
    setup_module = _load_py_module(name="registry_setup", location="src/test_registry/__setup__.py")
else:
    raise ValueError(f"Invalid package name {pkg_name}")
