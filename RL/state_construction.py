import torch
import torch.nn as nn
import pytorch_quantization.nn as quant_nn

def construct_state(net, net_name, preserve_out_c, preserve_in_c, flops_list, ori_flops_list, actions_p):

    # Create a matrix like how AMC does
    if "CNN1D_TrafficClassification" == net_name:
    
        # Easier to hardcode the height and width. Can either use torchsummary or manually calc everytime pooling
        heights = [1.] * len(actions_p)  #cnn1d #row always = 1
        widths = [1500., 500., 166., 83., 41.]
        
        if (type(preserve_out_c) is not list):
            preserve_out_c = preserve_out_c.tolist()    # after 2nd episode, it becomes numpy
        
        state_embedding = []
        i = 0

        for layer in net.modules():

            if isinstance(layer, quant_nn.QuantConv1d):
                
                reduced = 0
                rest = 0

                if i == 0:                      # first layer
                    reduced = 0
                    rest = ori_flops_list[i+1] - flops_list[i+1]
                elif i == (len(actions_p) -1):  # last layer
                    reduced = ori_flops_list[i-1] - flops_list[i-1]
                    rest = 0
                else:
                    reduced = ori_flops_list[i-1] - flops_list[i-1]
                    rest = ori_flops_list[i+1] - flops_list[i+1]
                    
                layer_info = [
                    (i+1),
                    preserve_out_c[i],
                    preserve_in_c[i],
                    heights[i],
                    widths[i],
                    layer.stride[0],
                    layer.kernel_size[0],
                    flops_list[i],  # no. of flops for this layer
                    reduced,        # no. of reduced flops in previous layer
                    rest,
                    actions_p[i],
                ]
                state_embedding.append(layer_info)

                i += 1
        
        state_embedding = torch.tensor(state_embedding)
        
        # Compute the min and max values for each column
        min_vals = torch.min(state_embedding, dim=0)[0]
        max_vals = torch.max(state_embedding, dim=0)[0]

        # Define a small epsilon value to prevent division by zero
        epsilon = 1e-10

        # Apply min-max scaling with epsilon to avoid division by zero
        state_embedding_normalized = (state_embedding - min_vals) / (max_vals - min_vals + epsilon)
        state_embedding_normalized = state_embedding_normalized.clone().detach()

        return state_embedding_normalized.to(torch.float32)
    
    else:
        raise NotImplementedError(f"Did not implement state_construction() for {net_name}")