import os

import torch

baseLoc = "C:/Users/jrp32/Documents/Cambridge University/Year III/Project"
projectLoc = f"{baseLoc}/projectFiles"
noIndices = 253401
noIndicesPlusTags = noIndices + 3
maxLengthSentence = 414
gloveWidth = 300
fileLoc = "/".join(os.path.abspath(__file__).split("\\")[:-2])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device:{device}")
