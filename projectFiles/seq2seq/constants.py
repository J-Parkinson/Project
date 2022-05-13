import os

import torch

noIndices = 253401
noIndicesPlusTags = noIndices + 3
maxLengthSentence = 414
teacher_forcing_ratio = 0.5
gloveWidth = 300
bertWidth = 768

fileLoc = "/".join(os.path.abspath(__file__).split("\\")[:-2])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device:{device}")
