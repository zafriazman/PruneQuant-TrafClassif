import copy
import torch.nn as nn
import torch.nn.utils.prune as prune
import pytorch_quantization.nn

def channel_pruning(net, preserve_ratio):
    '''
    :param net: DNN
    :param preserve_ratio: preserve rate (1 - pruning actions)
    :return: newnet (nn.Module): a newnet contain mask that help prune network's weight
    '''

    if not isinstance(net, nn.Module):
        print('Invalid input. Must be nn.Module')
        return
    newnet = copy.deepcopy(net)
    i=0
    
    for name, module in newnet.named_modules():
        if isinstance(module, pytorch_quantization.nn.QuantConv1d):
            
            #print(name)
            #print("before")
            #print(list(module.named_parameters()))

            prune.ln_structured(module, name='weight', amount=float(1-preserve_ratio[i]), n=2, dim=0)
            
            #print("after")
            #print(list(module.named_parameters()))
            #print(list(module.named_buffers()))

            i+=1

    return newnet


def prune_permenantly(net):

    for name, module in net.named_modules():

        if isinstance(module, pytorch_quantization.nn.QuantConv1d):
            
            module = prune.remove(module, name='weight')