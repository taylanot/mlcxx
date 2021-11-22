import setuptools
import os

# get required information from package
packageName="Mechanics"                                                         # name of the package
filePath=os.path.dirname(__file__)                                              # path of this file

# get version
versionFile=os.path.join(filePath,packageName,"_version.py")                    # version file
with open(versionFile,"r") as fileHandle:                                       # open version file
    exec(fileHandle.read())                                                     # -> get version number

# get package description
readmeFile=os.path.join(filePath,packageName,"README.rst")                      # README file
with open("README.rst", "r") as fileHandle:                                     # open README file
    readmeInfo = fileHandle.read()                                              # -> get README information


# setup
setuptools.setup(
    name=packageName,
    version=__version__,
    author="Ozgur Taylan Turan",
    python_requires='>=3.8',
)
