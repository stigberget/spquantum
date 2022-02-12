import os,sys

from setuptools import find_packages,setup

root_dir = os.path.dirname(os.path.abspath(__file__))

modules = ["grid","solver","tests","utils"]

for module in modules: 
    sys.path.append(os.path.join(root_dir,module))


setup(
    name = "SPQuantum",
    version = "0.1.0",
    packages=["grid","utils","tests"],
)