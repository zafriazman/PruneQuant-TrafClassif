import tqdm
import torch
import pytorch_quantization.nn as quant_nn
import pytorch_quantization.calib as calib

from pytorch_quantization import tensor_quant
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer

def prepare_perlayer(quantized_net, bitwidths):

    if quantized_net.__class__.__name__ == "CNN1D_TrafficClassification":
        
        layer_index  = 0
        no_of_module = 0

        for module in quantized_net.modules():
            if isinstance(module, quant_nn.QuantConv1d):
                no_of_module += 1        

        if len(bitwidths) != no_of_module:
            raise ValueError(f"Ensure action's array [{len(bitwidths)}] == number of layer to quantize [{no_of_module}]")

        for name, module in quantized_net.named_modules():
            if isinstance(module, quant_nn.QuantConv1d):
                
                # Use "Histogram" calibrator  for activations. "True" bcoz relu
                module._input_quantizer.num_bits     = bitwidths[layer_index]
                module._input_quantizer.unsigned     = False
                module._input_quantizer._calibrator  = calib.HistogramCalibrator(
                                                            num_bits=module._input_quantizer.num_bits, 
                                                            axis=module._input_quantizer.axis, 
                                                            unsigned=module._input_quantizer.unsigned)
                
                # Maintain default "Min-Max" calibrator for weights
                module._weight_quantizer.num_bits    = bitwidths[layer_index]
                module._weight_quantizer.unsigned    = False

                #print(f"{name} Module after:\n{module}")
                layer_index += 1

    else:
        raise NotImplementedError(f"Did not implement quantization flow on {quantized_net.__class__.__name__}")
    
    return quantized_net




def action_to_bitwidth(actions, b_min, b_max):

    if b_min not in [2, 4, 8] or b_max != 8:
        raise ValueError("b_min must be one of [2, 4, 8] and b_max must be 8.")
    
    bitwidths = []

    for a in actions:
        if b_min == 2:
            # min() is to ensure that the minimum value after rounding is 1 for b_min = 2
            # max() is to ensure the result does not exceed the maximum allowed bit width
            # if action is exactly 1, the calc would be to round (3.5), so bitwidth will be 16 which we dont want
            rounded_value = max(round(1 + (3 * a) - 0.5), 1)
            b = 2 ** rounded_value
            b = min(b, b_max)
        elif b_min == 4:
            b = 4 if a < 0.5 else 8
        else:
            b = b_max
        bitwidths.append(b)
    
    return bitwidths