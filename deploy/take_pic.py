import cv2
import argparse
from pathlib import Path
from PIL import Image
# from mtcnn import MTCNN
from datetime import datetime

from PIL import Image
import numpy as np
import os

parser = argparse.ArgumentParser(description='take a picture')
parser.add_argument('--name', '-n', default='unknown', type=str, help='input the name of the recording person')
args = parser.parse_args()

data_path = Path('data')
save_path = os.path.join(data_path, 'facebank' , args.name)
if not os.path.exists(save_path):
    os.makedirs(save_path)

import cv2
import time

cap=cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

while(cap.isOpened()):
    ret_flag, Vshow = cap.read()
    cv2.imshow('Capture', Vshow)
    k=cv2.waitKey(1)
    if k==ord('s'):
        save_file_name = os.path.join(save_path,'{}.jpg'.format(str(datetime.now())[:-7].replace(":", "-").replace(" ", "-")))
        cv2.imwrite(save_file_name,np.asarray(Vshow))
        print(cap.get(3))
        print(cap.get(4))
    elif k==ord('q'):
        print('完成')
        break
cap.release()
cv2.destoryAllWindows()
