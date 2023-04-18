import os
import SimpleITK as sitk
import shutil
from matplotlib import pyplot as plt

dir = r'F:\ExValNF1119out'
outDir = r'F:\ExValNF1119outSelected'
os.makedirs(outDir, exist_ok=True)

for file in os.listdir(dir):
    if file.endswith('.nii.gz'):
        inPath = os.path.join(dir, file)
        mask = sitk.GetArrayFromImage(sitk.ReadImage(inPath)).squeeze()
        if mask.sum() != 0:
            shutil.copy(inPath, outDir)
            print(file)