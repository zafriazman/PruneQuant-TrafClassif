import os
import sys
import time
import copy
import torch
import tensorrt
import torch_tensorrt
import torch_tensorrt.fx
import torch.nn as nn
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch.ao.quantization
import torch.ao.quantization.quantize_fx
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx, fuse_fx
import torch.onnx

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

    # Common quantization aware training (QAT) / post training quantization (PTQ) workflow in Pytorch
    #   (1) Minor_model_modification (e.g. fuse conv with bn or etc - This is automatically done in fx)
    #   (2) Prepare (Place observer at the part to quantize)
    #   (3) Calibrate (QAT calibrate by training dataset. This is to estimate activation's scale and zero_point)
    #   (4) Convert

    
    # Get dataset
    start_time  = time.time()
    device      = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_loader, valid_loader, test_loader, classes = create_data_loaders_iscx2016vpn(
        os.path.join('data', 'datasets', 'iscx2016vpn-pytorch'),
        batch_size  = 1024,
        n_worker    = 0,
        train_ratio = 0.65,
        val_ratio   = 0.15
    )
    example_inputs  = (next(iter(test_loader))[0]).to(device)
    criterion       = nn.CrossEntropyLoss()
    qat_epoch       = 1

    # Load pretrained floating point model
    net                 = load_fp32_model(input_ch=example_inputs.shape[1], num_classes=classes, device=device)
    quantized_net       = load_fp32_model(input_ch=example_inputs.shape[1], num_classes=classes, device=device)
    test_loss, test_acc = validate(net, test_loader, criterion, device)
    print(f"Acc on unseen test data (not validation dataset): {test_acc}")

    # Bitwidth perlayer (each conv1d is 1 layer)
    #action_array    = [1., 0., 0.5, 0.2, 0.9]
    action_array    = [1., 1., 1., 1., 1.]
    bmin            = 2
    bmax            = 8
    bitwidth_array  = action_to_bitwidth(action_array, bmin, bmax)
    
    # Prepare
    qconfig_mapping = prepare_qconfig_perlayer(quantized_net, bitwidth_array)
    quantized_net   = prepare_qat_fx(quantized_net, qconfig_mapping, example_inputs)

    # Calibrate
    calibrate(quantized_net, valid_loader, qat_epoch, device)

    # Convert
    quantized_net.eval()
    quantized_net.apply(torch.ao.quantization.disable_observer)
    quantized_net.to(torch.device("cpu"))
    quantized_net = convert_fx(quantized_net)

    # Evaluate converted quantized model on CPU (using convert_fx)
    start_time_inf  = time.time()
    top1, top5 = evaluate(quantized_net, criterion, test_loader)
    print(f"Using CPU: Acc on unseen test data (not validation dataset): {top1}")
    print(f"Using CPU: Elapsed time for 20% of dataset: {time.time() - start_time_inf:.2f} seconds")

    # Convert to TensorRT so that can infer of GPU instead of CPU

    ########################################################################################################
    # torch_tensorrt.fx FAILS
    """ start_time_inf  = time.time()
    quantized_net.to(device)
    example_inputs.to(device)
    
    import torch.jit
    quantized_net.eval()
    quantized_net.to(device)
    example_inputs.to(device)
    #scripted_model = torch.jit.script(quantized_net)
    scripted_model = torch.jit.trace(quantized_net.cpu(), example_inputs.cpu())
    scripted_model.to(device)
    quantized_net.to(device)
    example_inputs.to(device)


    #compile_spec = {"inputs": [torch_tensorrt.Input([1024, 1, 1500])],
    #                "ir": "default",
    #                "enabled_precisions": torch.int8,
    #}
    #trt_model = torch_tensorrt.compile(scripted_model, **compile_spec)

    import torch_tensorrt.ptq
    import torch_tensorrt.fx
    import torch_tensorrt.fx.utils
    from torch_tensorrt.fx.utils import LowerPrecision


    trt_model = torch_tensorrt.fx.compile(
        module=quantized_net.cpu(),
        input=[example_inputs.cpu()],
        lower_precision=LowerPrecision.INT8
    )

    print(f"Using CUDA (TensorRT): Elapsed time for 20% of dataset: {time.time() - start_time_inf:.2f} seconds") """
    ########################################################################################################
    # pytorch.convert_fx to ONNX to TensorRT
    # Failed at converting ONNX to TRT engine. Most likely reason is: it failed at ONNX conversion
    # because log is showing that ONNX model thinks the model is INT64
    # 
    # To run the trtexec in shell:
    # >>> echo $CONDA_PREFIX
    # /home/zafri/anaconda3/envs/cuda12
    # >>> export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/zafri/anaconda3/envs/cuda12/lib
    # /home/zafri/anaconda3/envs/cuda12/TensorRT-8.6.1.6/bin/trtexec --onnx=networks/quantized_models/iscx2016vpn/model.onnx --saveEngine=networks/quantized_models/iscx2016vpn/model_engine.trt
    # Need to unset the LD_LIBRARY_PATH to original. Or else cannot run the command

    start_time_inf  = time.time()
    import onnx
    import torch.onnx
    import onnxruntime as ort
    import subprocess
    # onnx dynamo
    #onnx_model = torch.onnx.dynamo_export(quantized_net.cpu(), example_inputs.cpu()) 
    # Fail using dynamo at quantize_per_tensor = torch.quantize_per_tensor(x, conv1_input_scale_0, conv1_input_zero_point_0, torch.quint8);

    # onnx script
    quantized_net.eval()
    onnx_path = os.path.join("networks","quantized_models","iscx2016vpn","model.onnx")
    trt_engine_path = os.path.join("networks","quantized_models","iscx2016vpn","model_engine.trt")
    torch.onnx.export(quantized_net.cpu(), args=example_inputs.cpu(), f=onnx_path,
                      export_params=True, verbose=False, input_names=['input'], output_names=['output'])
    model_onnx = onnx.load(onnx_path)
    #print(onnx.helper.printable_graph(model_onnx.graph))
    # set providers to ['TensorrtExecutionProvider', 'CUDAExecutionProvider'] with TensorrtExecutionProvider having the higher priority.
    #sess = ort.InferenceSession(onnx_path, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'])
    command = 'conda activate cuda12'   # all this is not working. This is not the correct way to run multiple lines of code.
    subprocess.run(command, shell=True)
    command = 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/zafri/anaconda3/envs/cuda12/lib'
    subprocess.run(command, shell=True)
    command = '/home/zafri/anaconda3/envs/cuda12/TensorRT-8.6.1.6/bin/trtexec --onnx=networks/quantized_models/iscx2016vpn/model.onnx --saveEngine=networks/quantized_models/iscx2016vpn/model_engine.trt'
    subprocess.run(command, shell=True)
    
    


    
    
    print(f"Using CUDA (TensorRT): Elapsed time for 20% of dataset: {time.time() - start_time_inf:.2f} seconds")
    ########################################################################################################


    


    

    print(f"Total elapsed time: {time.time() - start_time:.2f} seconds")

    