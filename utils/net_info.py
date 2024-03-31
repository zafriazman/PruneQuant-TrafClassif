from torch import nn
from networks.cnn1d import CNN1D_TrafficClassification


def get_num_hidden_layer(net, model_name):

    n_layer=0

    if model_name in "CNN1D_TrafficClassification":
        for name, module in net.named_modules():
            if isinstance(module, nn.Conv1d):
                n_layer += 1
    else:
        raise NotImplementedError
    
    return n_layer


def get_num_in_out_channel(net, model_name):

    in_channels  = []
    out_channels = []

    if model_name == "CNN1D_TrafficClassification":
    
        for name, layer in net.named_modules():
            if isinstance(layer, nn.Conv1d):
                in_channels.append(layer.in_channels)
                out_channels.append(layer.out_channels)
    else:
        raise NotImplementedError(f"Did not implement get_num_in_out_channel for {model_name}")
    
    return in_channels, out_channels