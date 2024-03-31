import os
import sys
import onnx
import copy
import time
import torch
import torch.onnx
import torch.nn as nn
import pytorch_quantization
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pytorch_quantization import quant_modules
from pytorch_quantization import nn as quant_nn
from networks.cnn1d import CNN1D_TrafficClassification
from utils.iscx2016vpn_training_utils import create_data_loaders_iscx2016vpn
from prune_quant.prune import channel_pruning, prune_permenantly, workaround_deepcopy
from prune_quant.quant_prepare import action_to_bitwidth, prepare_perlayer
from prune_quant.quant_calibrate import compute_amax, collect_stats, qat_train
from prune_quant.quant_convert import convert_to_onnx
from prune_quant.quant_utils import load_fp32_model, save_model_state_dict, benchmark_against_NiN



def main_qat_flow(args, model_bckup, train_loader, val_loader, num_classes, preserve_array_prune, n_layer):

    start_time     = time.time()
    device         = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size     = args.data_bsize
    example_inputs = (next(iter(val_loader))[0]).to(device)
    criterion      = nn.CrossEntropyLoss()
    qat_epoch      = args.qat_epochs
    bitwidth_array = [8] * n_layer
    model          = copy.deepcopy(model_bckup)

    # Any conv/linear/pool layer that is declared after this code, will be substituted with quant version of it (via monkey patching)
    quant_modules.initialize()
    if model.__class__.__name__ == "CNN1D_TrafficClassification":
        quantized_net = CNN1D_TrafficClassification(input_ch=example_inputs.shape[1], num_classes=num_classes).to(device)
        quantized_net.load_state_dict(model.state_dict())

    quantized_net = prepare_perlayer(quantized_net, bitwidth_array)

    with torch.no_grad():
        collect_stats(quantized_net, train_loader, num_batches=(1000//batch_size))
        compute_amax(quantized_net, method="percentile", percentile=99.99)
    quantized_net.cuda()

    # Pruning - add prune mask
    prune_quant_net = channel_pruning(net=quantized_net, preserve_ratio=preserve_array_prune)

    # Train model
    qat_train(prune_quant_net, train_loader, qat_epoch, device, args.lr, args.qat_momentum, args.qat_wd)

    # Create a copy with masks, so that can finetune later
    prune_quant_net, prune_quant_net_w_mask = workaround_deepcopy(prune_quant_net, n_layer)

    # Remove pruning masks to prune permenantly
    prune_permenantly(prune_quant_net)

    return prune_quant_net, prune_quant_net_w_mask




if __name__ == "__main__":

    # Get dataset
    start_time      = time.time()
    device          = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size      = 1
    n_worker        = 0
    train_ratio     = 0.65
    val_ratio       = 0.15
    dataset         = os.path.join('data', 'datasets', 'iscx2016vpn-pytorch')
    train_loader, valid_loader, test_loader, classes = create_data_loaders_iscx2016vpn(dataset, batch_size, n_worker, train_ratio, val_ratio)
    
    
    # Set hyperparameters
    example_inputs        = (next(iter(test_loader))[0]).to(device)
    criterion             = nn.CrossEntropyLoss()
    qat_epoch             = 1
    action_array_quant    = [1., 1. , 1., 1., 1.]
    preserve_array_prune  = [1., 0.9, 1., 1., 1.]
    bmin                  = 2
    bmax                  = 8
    bitwidth_array        = action_to_bitwidth(action_array_quant, bmin, bmax)


    # Load pretrained floating point model
    path_own_model = os.path.join('networks', 'pretrained_models', 'iscx2016vpn', 'CNN1D_TrafficClassification_best_model_without_aux.pth')
    path_NiN_model = os.path.join('networks', 'pretrained_models', 'iscx2016vpn', 'NiN_CNN1D_TrafficClassification_best_model_without_aux.pth')
    net            = load_fp32_model(path=path_own_model, input_ch=example_inputs.shape[1], num_classes=classes, device=device)
    net_NiN        = load_fp32_model(path=path_NiN_model, input_ch=example_inputs.shape[1], num_classes=classes, device=device)


    # Create quantized model. 
    # Any conv/linear/pool layer that is declared after this code, will be substituted with quant version of it (via monkey patching)
    quant_modules.initialize()
    quantized_net = load_fp32_model(path=path_own_model, input_ch=example_inputs.shape[1], num_classes=classes, device=device)
    model_name = quantized_net.__class__.__name__

    
    # Prepare
    quantized_net = prepare_perlayer(quantized_net, bitwidth_array)


    # Calibrate
    with torch.no_grad():
        collect_stats(quantized_net, train_loader, num_batches=(1000//batch_size))
        compute_amax(quantized_net, method="percentile", percentile=99.99)
    quantized_net.cuda()
    
    
    # Pruning - add prune mask
    prune_quant_net = channel_pruning(net=quantized_net, preserve_ratio=preserve_array_prune)


    # Train and save the model (with masks)
    qat_train(quantized_net=prune_quant_net, data_loader=train_loader, epochs=qat_epoch, device=device, lr=0.005, qat_momentum=0.9, qat_wd=4e-5)
    save_model_state_dict(prune_quant_net, model_name, "iscx2016vpn", preserve_array_prune, bitwidth_array)


    # Remove pruning masks to prune permenantly
    prune_permenantly(prune_quant_net)


    # Convert to onnx
    onnx_path       = os.path.join("networks","quantized_models","iscx2016vpn","model.onnx")
    trt_engine_path = os.path.join("networks","quantized_models","iscx2016vpn","model_engine.trt")
    convert_to_onnx(prune_quant_net, example_inputs, onnx_path)


    # Benchmark the trt
    benchmark_against_NiN(net, net_NiN, trt_engine_path, test_loader, batch_size, classes, criterion, device)
    print("Note that this benchmark is using previous trt file. Perform trtexec to convert .onnx to .trt engine, then rerun this script")
    print(f"Total elapsed time: {time.time() - start_time:.2f} seconds")

    