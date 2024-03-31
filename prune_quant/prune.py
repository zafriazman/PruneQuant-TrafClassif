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


def workaround_deepcopy(pruned_net, n_layer):
    '''
    For deepcopy pruned model with masks. Take pruned_net with masks, output 2 pruned_net with masks
    Needed because of this error:
    RuntimeError: Only Tensors created explicitly by the user (graph leaves) support the deepcopy protocol at the moment.
    '''
    state_dict_w_mask = pruned_net.state_dict()
    prune_permenantly(pruned_net)

    copymodel = copy.deepcopy(pruned_net)
    copymodel = channel_pruning(copymodel, [1.]*n_layer)
    copymodel.load_state_dict(state_dict_w_mask)

    pruned_net = channel_pruning(pruned_net, [1.]*n_layer)
    pruned_net.load_state_dict(state_dict_w_mask)

    return pruned_net, copymodel



