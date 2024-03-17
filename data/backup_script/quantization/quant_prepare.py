import torch
import torch.nn as nn
import torch.ao.quantization
from torch.ao.quantization import QConfig
from torch.ao.quantization import QConfigMapping


def prepare_qconfig(name, no_of_bit):

    #print("Name: {:20s} \t| #bit: {:2d} \t| Activation quant_max: {:3d} \t| Weight quant_min: {:3d} \t| Weight quant_max: {:3d}" 
    #      .format(name[:20], int(no_of_bit), int(2**no_of_bit-1), int(-(2**no_of_bit)/2), int((2**no_of_bit)/2-1)))

    # for activation_function qconfig
    # most accurate observer is histogram, but too slow, therefore used MinMaxObs.
    # In QAT MovingAverageMinMaxObserver is used to gradually alter the range as the weights change
    # this quant_min/max will control the range of the Bitwidth
    # But PyTorch only support 8 bit quantization. So expect no further storage saving for 4 or 2 bit
    # eg what will happen if set 4 bit range: |-|-|-|-|1|0|1|0|
    # Will only change last 4 bits, but still store in 8bit memory
    # affine offer tighter clipping ranges and are useful for quantizing non-negative activations. Since ReLu remove -ve no.
    # Reduce range is for real HW overflow problem on x86 . Even though we set quint8(8 bit), only 7 bit can actually be used
    # For tensorrt, no need to reduce range
    activation_qconfig = torch.ao.quantization.fake_quantize.FakeQuantize.with_args(
        observer=torch.ao.quantization.observer.MinMaxObserver,
        quant_min=int(0),
        quant_max=int(2**no_of_bit-1),
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False)
    
    # for weight qconfig
    # This example shows if we use signed 8 bit integer (above is unsigned)
    # For weights, make it symmetric at 0, thus no need calc zero_point
    # This is common configuration from survey paper (activation: affine bcoz relu, weight: symmetry bcoz dont want to calc zero_point)
    weight_qconfig = torch.ao.quantization.fake_quantize.FakeQuantize.with_args(
        observer=torch.ao.quantization.observer.MinMaxObserver,
        quant_min=int(-(2**no_of_bit)/2),
        quant_max=int((2**no_of_bit)/2-1),
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        reduce_range=False)
    
    return QConfig(activation=activation_qconfig,  weight=weight_qconfig)


def prepare_qconfig_perlayer(net, bitwidths):

    if net.__class__.__name__ == "CNN1D_TrafficClassification":
        
        no_of_module = 0
        layer_index = 0
        #qconfig_mapping = QConfigMapping().set_global(get_default_qat_qconfig_mapping("x86"))
        qconfig_mapping = (QConfigMapping()
                           .set_global(prepare_qconfig("Global", int(8)))
                           .set_module_name("conv1", prepare_qconfig("conv1", bitwidths[0]))
                           .set_module_name("conv2", prepare_qconfig("conv2", bitwidths[1]))
                           .set_module_name("conv3", prepare_qconfig("conv3", bitwidths[2]))
                           .set_module_name("conv4", prepare_qconfig("conv4", bitwidths[3]))
                           .set_module_name("conv5", prepare_qconfig("conv5", bitwidths[4]))
        )

        for module in net.modules():
            if isinstance(module, nn.Conv1d):
                no_of_module += 1
        
        """ # Eager mode approach
        for name, module in net.named_modules():
            if isinstance(module, nn.Conv1d):
                activation_qconfig, weight_qconfig = prepare_qconfig(name, no_of_bit=bitwidths[layer_index])
                module.qconfig = torch.ao.quantization.QConfig(activation=activation_qconfig, weight=weight_qconfig)
                layer_index += 1 """
        
                
        if len(bitwidths) != no_of_module:
            print("Ensure action's array == number of layer to quantize")
            print("action's array:", bitwidths, "\tnumber of layer to quantize:", no_of_module)
            exit()

    else:
        raise NotImplementedError(f"Did not implement quantization flow on {net.__class__.__name__}")
    
    return qconfig_mapping


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