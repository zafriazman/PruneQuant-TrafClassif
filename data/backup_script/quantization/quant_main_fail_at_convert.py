import os
import sys
import time
import copy
import torch
import tensorrt
import torch.nn as nn
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch.ao.quantization
import torch.ao.quantization.quantize_fx
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx, fuse_fx

from tqdm import tqdm
from torch import optim
from torch.nn.utils import prune
from collections import OrderedDict
from networks.cnn1d import CNN1D_TrafficClassification
from torch.utils.data import DataLoader, SubsetRandomSampler
from utils.iscx2016vpn_training_utils import create_data_loaders_iscx2016vpn, validate
from quantization.quant_utils import AverageMeter, accuracy, evaluate, print_size_of_model, load_fp32_model
from quantization.quant_prepare import prepare_qconfig_perlayer, action_to_bitwidth
from quantization.quant_calibrate import calibrate


if __name__ == "__main__":

    # Common quantization aware training (QAT) / post training quantization (PTQ) workflow 
    #   (1) Minor_model_modification
    #   (2) Prepare (Place observer at the part to quantize)
    #   (3) Calibrate (QAT calibrate by training dataset. This is to estimate activation's scale and zero_point)
    #   (4) Convert

    
    # Get dataset
    start_time = time.time()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_loader, valid_loader, test_loader, classes = create_data_loaders_iscx2016vpn(
        os.path.join('data', 'datasets', 'iscx2016vpn-pytorch'),
        batch_size=1024,
        n_worker=0,
        train_ratio=0.65,
        val_ratio=0.15
    )
    example_inputs = (next(iter(test_loader))[0])
    criterion  = nn.CrossEntropyLoss()
    qat_epoch = 1

    # Load pretrained floating point model
    net = load_fp32_model(input_ch=example_inputs.shape[1], num_classes=classes, device=device)
    quantized_net = load_fp32_model(input_ch=example_inputs.shape[1], num_classes=classes, device=device)
    #test_loss, test_acc = validate(net, test_loader, criterion, device)
    #print(f"Acc on unseen test data (not validation dataset): {test_acc}")

    # Bitwidth perlayer (each conv1d is 1 layer)
    #action_array = [1., 0., 0.5, 0.2, 0.9]
    action_array = [1., 1., 1., 1., 1.]
    bmin = 2; bmax = 8
    bitwidth_array = action_to_bitwidth(action_array, bmin, bmax)
    
    # Prepare
    #qconfig_mapping = prepare_qconfig_perlayer(quantized_net, bitwidth_array)
    qconfig_mapping = torch.ao.quantization.get_default_qat_qconfig_mapping("qnnpack")
    
    quantized_net = prepare_qat_fx(quantized_net, qconfig_mapping, example_inputs)

    # Calibrate
    calibrate(quantized_net, valid_loader, qat_epoch, device)

    quantized_net.eval()
    quantized_net.to(torch.device("cpu"))

    # Convert
    quantized_net.apply(torch.ao.quantization.disable_observer)
    quantized_net.to(torch.device("cpu"))
    quantized_net = convert_fx(quantized_net)
    print(net)
    print("\n\n")
    print(quantized_net)    # Fail here (Segmentation fault)

    
    # Evaluate converted quantized model
    #top1, top5 = evaluate(quantized_net, criterion, test_loader)


    


    

    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")

    