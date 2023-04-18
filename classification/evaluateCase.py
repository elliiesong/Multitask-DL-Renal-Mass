import os
import pandas as pd


if __name__ == '__main__':
    predDir = r'F:\renal_cyst\src\Detector\Classifier\preds\cases'
    for file in os.listdir(predDir):
        df = pd.read_csv(predDir, file)
