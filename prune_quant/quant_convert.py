import os
import torch
import torch.onnx
import pytorch_quantization


def convert_to_onnx(prune_quant_net, example_inputs, onnx_path):

    prune_quant_net.eval()
    prune_quant_net.cuda()
    example_inputs.cuda()
    
    with pytorch_quantization.enable_onnx_export():  
        torch.onnx.export(prune_quant_net, example_inputs, onnx_path, verbose=False, opset_version=17, do_constant_folding=True)

    #model_onnx = onnx.load(onnx_path)
        
    # my Shell command to run trtexec for converting .onnx to .trt engine
    cmd1 = 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/zafri/anaconda3/envs/cuda12:/home/zafri/anaconda3/envs/cuda12/lib'
    cmd2 = '/home/zafri/anaconda3/envs/cuda12/TensorRT-8.6.1.6/bin/trtexec --int8 --onnx=networks/quantized_models/iscx2016vpn/model.onnx --saveEngine=networks/quantized_models/iscx2016vpn/model_engine.trt'