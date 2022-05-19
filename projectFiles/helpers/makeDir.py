import os

from projectFiles.constants import projectLoc


def makeDir(fileName, startDir=f"{projectLoc}", full=False):
    os.mkdir(f"{startDir}/{fileName}")
    if full:
        return f"{startDir}/{fileName}"
    return fileName
