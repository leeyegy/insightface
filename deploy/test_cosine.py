import face_model
import argparse
import cv2
import sys
import numpy as np
from mxnet2pytorch import *
from model_pytorch import *
import torch
import torch.nn.functional as F
import  os


parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='../models/model-r100-ii/model,0', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
parser.add_argument('--jiang_img', default="", type=str, help='ver dist threshold')
args = parser.parse_args()

# define image base path
base_path = os.path.join("data","Baseline-10-9")


def cosine_dist(emb1, emb2):
    return (1. - F.cosine_similarity(emb1, emb2)).sum()

#load model
## mxnet version
model = face_model.FaceModel(args)
## pytorch version
r100 = Backbone(num_layers=100, drop_ratio=0.4, mode='ir')
transfer = Mxnet2Pytorch()
pytorch_model = transfer.init_model(r100, args.model).eval()


# generate jiang feature
jiang_path = os.path.join(base_path,"jiang",args.jiang_img)
jiang = cv2.imread(jiang_path)
jiang = model.get_input(jiang)
jiang_feature = pytorch_model(torch.unsqueeze(torch.from_numpy(jiang),0))


# traverse target list
wu_dir = os.path.join(base_path,"jiang")
list = os.listdir(wu_dir) #列出文件夹下所有的目录与文件
for i in range(0,len(list)):
       wu_path = os.path.join(wu_dir,list[i])
       if os.path.isfile(wu_path):
           wu = cv2.imread(wu_path)
           wu = model.get_input(wu)
           wu_feature = pytorch_model(torch.unsqueeze(torch.from_numpy(wu), 0))
           print("图片{}与图片:{} 的余弦距离为:{}".format(jiang_path,wu_path,cosine_dist(wu_feature,jiang_feature)))
