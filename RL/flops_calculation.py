import os
import sys
import copy
import torch
import torch.nn as nn
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from networks.cnn1d import CNN1D_TrafficClassification
from networks.cnn1d_NiN import NiN_CNN1D_TrafficClassification


def calc_perlayer_flops(orig_model, preserve_ratios):

    # Formula is (2 * {c_in * [c_out * preserve_ratio] * k_height * k_width} + [c_out * preserve_ratio]) * h_out_height * h_out_width
    # the 2 * {} is because 1 MAC is 2 FLOPs (1 multiply & 1 addition)
    # c_in (input channel) is equal to each kernel's number of channel (eg. 224x224x3 image can only mul with _x_x3 kernel to produce 1 out channel)
    # c_out is equal to the number of kernel (eg. if input is 224x224x3, and we want to get 224x224x64 output, we need 64 of _x_x3 kernel)
    # preserve_ratio of baseline(not pruned) model should be set to a list of 1.0s
    # the addition (...} + [c_out...) is for the bias. Since each kernel will have 1 bias
    # 1500 is the number of input's features, which is the maximum transmission unit

    model = copy.deepcopy(orig_model)
    output_perlayers = None
    flops_list = []
    i = 0

    if model.__class__.__name__ == "CNN1D_TrafficClassification":
        output_perlayers = [1500, 500, 166, 83, 41]  # either use torchsummary or manually calc based on pooling ops

    elif model.__class__.__name__ == "NiN_CNN1D_TrafficClassification":
        output_perlayers = [750, 750, 750, 375, 375, 375, 187, 187, 187, 187]

    else:
        raise NotImplementedError(f"Did not implement calc_perlayer_flops for {model.__class__.__name__}")

    for name, module in model.named_modules():

        #if isinstance(module, nn.Conv1d):  # Not sure why this does not work even though type(module) is clearly nn.Conv1d
        if module.__class__.__name__ == 'Conv1d':

            c_in = module.in_channels
            c_out = module.out_channels
            k_height = 1
            k_width = module.kernel_size[0]
            h_out_height  = 1
            h_out_width = output_perlayers[i]

            layer_flops = (2 * (c_in * (c_out * preserve_ratios[i]) * k_height * k_width) + (c_out * preserve_ratios[i])) * h_out_height * h_out_width
            flops_list.append(layer_flops)

            i += 1
    
    return flops_list



if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = CNN1D_TrafficClassification(input_ch=1500, num_classes=16).to(device)
    flops_list = calc_perlayer_flops(net, [1]*5)
    print(f"My model FLOPs: {sum(flops_list)}\t {flops_list}")

    net = NiN_CNN1D_TrafficClassification(input_ch=1500, num_classes=16).to(device)
    flops_list = calc_perlayer_flops(net, [1]*10)
    print(f"NiN model FLOPs: {sum(flops_list)}\t {flops_list}")


