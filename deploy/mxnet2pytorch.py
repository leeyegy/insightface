import face_model
import argparse
import cv2
import sys
import numpy as np
from model_pytorch import *
import mxnet as mx

# model initialization for PyTorch from MXnet params
class Mxnet2Pytorch(object):
    def __init__(self):
        pass
    def init_model(self, pytorch_model, mxnet_model_path):
        sym, param_dict, aux_params = self.get_model(mxnet_model_path)
        # print(pytorch_model)
        for n, m in pytorch_model.named_modules():
            if isinstance(m, BatchNorm2d) or isinstance(m,BatchNorm1d):
                # print(n)
                self.bn_init(n, m, param_dict, aux_params)
            elif isinstance(m, Conv2d):
                # print(n)
                self.conv_init(n, m, param_dict)
            elif isinstance(m, Linear):
                # print(n)
                self.fc_init(n, m, param_dict)
            elif isinstance(m, PReLU):
                # print(n)
                self.prelu_init(n, m, param_dict)

        return pytorch_model

    def _get_stage_and_unit(self,name):
        spl = name.split(".")  # 1,3
        index = int(spl[1])
        if index>= 46:
            stage = 4
            unit  = (index-46) + 1
        elif index >=16:
            stage = 3
            unit = (index-16) +1
        elif index >=3:
            stage = 2
            unit = (index-3) +1
        else:
            stage = 1
            unit = index +1
        return stage,unit

    def _get_bn_index(self,name):
        spl = name.split(".")  # 1,3
        index = int(spl[3])
        if index== 0:
            return 1
        elif index ==2 :
            return 2
        elif index ==5:
            return 3

    def _get_conv_index(self,name):
        spl = name.split(".")  # 1,3
        index = int(spl[3])
        if index== 1:
            return 1
        elif index ==4 :
            return 2

    def bn_init(self, n, m, param_dict, aux_params):
        # beta and gamma
        if not (m.weight is None):
            if "input" in n:
                m.weight.data.copy_(torch.FloatTensor(param_dict['bn0_gamma'].asnumpy()))
                m.bias.data.copy_(torch.FloatTensor(param_dict['bn0_beta'].asnumpy()))
            elif "output" in n:
                m.weight.data.copy_(torch.FloatTensor(param_dict['bn1_gamma' if "0" in n else "fc1_gamma"].asnumpy()))
                m.bias.data.copy_(torch.FloatTensor(param_dict['bn1_beta' if "0" in n else "fc1_beta"].asnumpy()))
            elif "body" in n:
                stage, unit = self._get_stage_and_unit(n)
                bn_index = self._get_bn_index(n)
                if "shortcut" in n :
                    m.weight.data.copy_(
                        torch.FloatTensor(param_dict["stage{}_unit{}_sc_gamma".format(stage,unit)].asnumpy()))
                    m.bias.data.copy_(torch.FloatTensor(param_dict["stage{}_unit{}_sc_beta".format(stage,unit)].asnumpy()))
                else:
                    m.weight.data.copy_(
                        torch.FloatTensor(param_dict["stage{}_unit{}_bn{}_gamma".format(stage, unit,bn_index)].asnumpy()))
                    m.bias.data.copy_(torch.FloatTensor(param_dict["stage{}_unit{}_bn{}_beta".format(stage, unit,bn_index)].asnumpy()))
        # moving mean and moving var
        if "input" in n:
            m.running_mean.copy_(torch.FloatTensor(aux_params['bn0_moving_mean'].asnumpy()))
            m.running_var.copy_(torch.FloatTensor(aux_params['bn0_moving_var'].asnumpy()))
        elif "output" in n:
            m.running_mean.copy_(torch.FloatTensor(aux_params['bn1_moving_mean' if "0" in n else "fc1_moving_mean"].asnumpy()))
            m.running_var.copy_(torch.FloatTensor(aux_params['bn1_moving_var' if "0" in n else "fc1_moving_var"].asnumpy()))
        elif "body" in n:
            # calculate the stage and unit
            stage,unit = self._get_stage_and_unit(n)
            bn_index = self._get_bn_index(n)
            if "shortcut" in n:
                m.running_mean.copy_(torch.FloatTensor(aux_params['stage{}_unit{}_sc_moving_mean'.format(stage,unit)].asnumpy()))
                m.running_var.copy_(torch.FloatTensor(aux_params['stage{}_unit{}_sc_moving_var'.format(stage,unit)].asnumpy()))
            else:
                m.running_mean.copy_(torch.FloatTensor(aux_params['stage{}_unit{}_bn{}_moving_mean'.format(stage,unit,bn_index)].asnumpy()))
                m.running_var.copy_(torch.FloatTensor(aux_params['stage{}_unit{}_bn{}_moving_var'.format(stage,unit,bn_index)].asnumpy()))

    def conv_init(self, n, m, param_dict):
        if "input" in n :
            m.weight.data.copy_(torch.FloatTensor(param_dict['conv0_weight'].asnumpy()))
        elif "output" in  n:
            print("error!")
            raise
            # m.weight.data.copy_(torch.FloatTensor(param_dict['conv0_weight'].asnumpy()))
        elif "body" in n :
            stage,unit = self._get_stage_and_unit(n)
            conv_index = self._get_conv_index(n)
            if "shortcut" in n:
                m.weight.data.copy_(torch.FloatTensor(param_dict['stage{}_unit{}_conv1sc_weight'.format(stage,unit)].asnumpy()))
            else:
                m.weight.data.copy_(torch.FloatTensor(param_dict['stage{}_unit{}_conv{}_weight'.format(stage,unit,conv_index)].asnumpy()))

        # no bias here
        # if n in ['conv1_1', 'conv4_1', 'conv3_1', 'conv2_1']:
        #     m.bias.data.copy_(torch.FloatTensor(param_dict[n + '_bias'].asnumpy()))

    def fc_init(self, n, m, param_dict):
        print("linear recovery!")
        # print(n)
        m.weight.data.copy_(torch.FloatTensor(param_dict['pre_fc1_weight'].asnumpy()))
        m.bias.data.copy_(torch.FloatTensor(param_dict['pre_fc1_bias'].asnumpy()))

    def prelu_init(self, n, m, param_dict):
        if "input" in n :
            m.weight.data.copy_(torch.FloatTensor(param_dict['relu0_gamma'].asnumpy()))
        elif "output" in  n:
            print("error!")
            raise
        elif "body" in n :
            stage,unit = self._get_stage_and_unit(n)
            if "shortcut" in n:
                print("error!")
                raise
            else:
                m.weight.data.copy_(torch.FloatTensor(param_dict['stage{}_unit{}_relu1_gamma'.format(stage,unit)].asnumpy()))

    def get_model(self,model_path):
        _vec = model_path.split(',')
        print(_vec)
        assert len(_vec) == 2
        prefix = _vec[0]
        epoch = int(_vec[1])
        print('loading', prefix, epoch)
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        return sym, arg_params,aux_params

def get_model(model_path):
    _vec = model_path.split(',')
    print(_vec)
    assert len(_vec) == 2
    prefix = _vec[0]
    epoch = int(_vec[1])
    print('loading', prefix, epoch)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = sym.get_internals()
    sym = all_layers[layer + '_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    # model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
    model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    return sym, arg_params,aux_params,model

if __name__ == "__main__":
    r100 = Backbone (num_layers=100, drop_ratio=0.4, mode='ir')
    transfer = Mxnet2Pytorch()
    transfer.init_model(r100,"../models/model-r100-ii/model,0")