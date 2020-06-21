import cv2
from tqdm import tqdm
import numpy as np
import os

BASE_DIR = ""
source = os.path.join(BASE_DIR, "SourceFolder")
save = os.path.join(BASE_DIR, "SaveFolder")

SCALE_PERCENT = 60


# Creat folders + Down scale images
def transform(source, savesource):
    files = os.listdir(source)
    files = np.asarray(files)
    for i in tqdm(range(len(files))):
        link = os.path.join(source, files[i])
        img = cv2.imread(link)
        width = int(img.shape[1] * SCALE_PERCENT / 100)
        height = int(img.shape[0] * SCALE_PERCENT / 100)
        dim = (width, height)
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(savesource, files[i]), resized)
