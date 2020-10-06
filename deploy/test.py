import face_model
import argparse
import cv2
import sys
import numpy as np
from mxnet2pytorch import *
from model_pytorch import *
import torch

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

#load model
## mxnet version
model = face_model.FaceModel(args)
## pytorch version
r100 = Backbone(num_layers=100, drop_ratio=0.4, mode='ir')
transfer = Mxnet2Pytorch()
pytorch_model = transfer.init_model(r100, args.model).eval()

img_ = cv2.imread('Tom_Hanks_54745.png')
img_ = model.get_input(img_)
f1 = model.get_feature(img_)
f1pytorch = pytorch_model(torch.unsqueeze(torch.from_numpy(img_),0))
print("Pytorch与MXNET输出平均处理差异：")
print ((torch.from_numpy(f1)-f1pytorch).abs().mean().item())
print("Pytorch与MXNET输出累计处理差异：")
print ((torch.from_numpy(f1)-f1pytorch).abs().sum().item())

img = cv2.imread('zmy.jpg')
img = model.get_input(img)
f2 = model.get_feature(img)
f2pytorch = pytorch_model(torch.unsqueeze(torch.from_numpy(img),0))

dist = np.sum(np.square(f1-f2))
print("mxnet下两张人脸的二范数距离：")
print(dist)
print("转成pytorch后两张人脸的二范数距离：")
print(((f1pytorch-f2pytorch)**2).sum().item())
sim = np.dot(f1, f2.T)
print("mxnet下两张人脸矩阵的相似度：")
print(sim)
print("转成pytorch后两张人脸矩阵的相似度：")
print(np.dot(f1pytorch.clone().detach().numpy(),f2pytorch.clone().detach().numpy().T))


#diff = np.subtract(source_feature, target_feature)
#dist = np.sum(np.square(diff),1)



