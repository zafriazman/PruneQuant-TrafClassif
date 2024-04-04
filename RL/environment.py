import os
import sys
import time
import copy
import torch
import numpy as np
import torch.nn as nn
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from parameter import parse_args
from utils.net_info import get_num_in_out_channel
from RL.state_construction import construct_state
from RL.flops_calculation import calc_perlayer_flops
from prune_quant.main import main_qat_flow
from prune_quant.quant_convert import convert_to_onnx
from prune_quant.quant_utils import evaluate, save_model_state_dict




class RL_env:

    def __init__(self, model, n_layer, dataset, train_loader, val_loader, test_loader, num_classes,
                 compression_ratio, state_dim, input_x, max_timesteps, model_name, device):

        # work space
        self.device  = device

        # DNN
        self.model                         = model
        self.model_name                    = model_name
        self.pruned_model                  = None
        self.input_x                       = input_x
        self.n_layer                       = n_layer
        self.flops                         = calc_perlayer_flops(self.model, [1.0]*self.n_layer)
        self.total_flops                   = sum(self.flops)
        self.total_bops                    = (self.total_flops/2) * 32 * 32
        self.in_channels,self.out_channels = get_num_in_out_channel(self.model, self.model_name)
        self.preserve_in_c                 = copy.deepcopy(self.in_channels)
        self.preserve_out_c                = copy.deepcopy(self.out_channels)
        self.pruned_out_c                  = None

        # dataset
        self.dataset      = dataset
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader
        self.num_classes  = num_classes

        # pruning and quantization
        self.desired_flops    = self.total_flops * compression_ratio
        self.desired_bops     = self.total_bops * compression_ratio
        self.preserve_ratio_p = torch.ones([n_layer])
        self.preserve_ratio_q = torch.ones([n_layer])
        self.best_accuracy    = 0

        # State
        self.state_dim      = state_dim
        self.current_states = None

        # env
        self.done          = False
        self.max_timesteps = max_timesteps
        _, accuracy, _     = evaluate(self.model, test_loader, nn.CrossEntropyLoss(), device)
        print(f"Initial val. accuracy on unseen test data (not validation dataset): {accuracy:.4f}%")


    def reset(self):
        self.done           = False
        self.pruned_model   = None
        self.preserve_ratio = torch.ones([self.n_layer])
        self.current_states = self.model_to_state(self.flops, self.flops, [0.]*self.n_layer)
        self.preserve_in_c  = copy.deepcopy(self.in_channels)
        self.preserve_out_c = copy.deepcopy(self.out_channels)
        self.pruned_out_c   = None
        return self.current_states


    def step(self, actions, time_step):
        
        start_time = time.time()
        args       = parse_args()
        rewards    = 0
        accuracy   = 0

        actions_p = actions
        self.preserve_ratio_p *= 1 - np.array(actions_p).astype(float)

        
        # Clip to prevent the whole layer to be pruned
        self.preserve_ratio_p = np.clip(self.preserve_ratio_p, 0.1, 1)

        self.calc_pruned_channels()

        # calc no. of flops in current step to find out how much flops has been reduced.
        current_flops = calc_perlayer_flops(self.model, self.preserve_ratio_p)
        reduced_flops = self.total_flops - sum(current_flops)

        if reduced_flops >= self.desired_flops:

            r_flops   = 1 - reduced_flops/self.total_flops
            self.done = True
           
            self.prune_quant_net, self.prune_quant_net_w_mask = main_qat_flow(args, self.model, self.train_loader, self.val_loader, self.num_classes, self.preserve_ratio_p, self.n_layer)

            loss, accuracy, avg_time = evaluate(self.prune_quant_net, self.val_loader, nn.CrossEntropyLoss(), self.device)
            rewards = accuracy
            
            if accuracy > self.best_accuracy:

                self.best_accuracy = accuracy

                save_model_state_dict(self.prune_quant_net_w_mask, self.model_name , self.dataset, self.preserve_ratio_p, ([8]*self.n_layer))
                if args.dataset == "iscx2016vpn":
                    onnx_path = os.path.join("networks","quantized_models","iscx2016vpn","model.onnx")
                elif args.dataset == "ustctfc2016":
                    onnx_path = os.path.join("networks","quantized_models","ustc-tfc2016","model.onnx")
                convert_to_onnx(self.prune_quant_net, self.input_x, onnx_path)
                print(f"Saving best model in onnx at {onnx_path}")
                print(f"Best accuracy is {accuracy:.2f}% with FLOPs ratio of {(r_flops*100):3f}% from baseline model")
                

        if time_step == (self.max_timesteps):
            if not self.done:
                rewards = 0
                self.done = True

        current_state = self.model_to_state(current_flops, self.flops, actions_p)
        #print("Step duration: %s seconds" % (time.time() - start_time))
        return current_state, rewards, self.done


    def calc_pruned_channels(self):
        '''
        Determine the number of pruned channels for each layer based on a given pruning ratio/percentage
        '''
        # [1:] means leave out the first layer. [-1] means leave out the last element in "preserve_ratio".
        # Each layer's input channel multiplied by the preserve ratio of "previous" layer
        # "previous" is used, because when we prune, we are pruning the kernel, not the input
        # out_channel is calculated normally

        self.preserve_in_c = copy.deepcopy(self.in_channels)
        self.preserve_in_c[1:] = (self.preserve_in_c[1:]*np.array(self.preserve_ratio_p[:-1]).reshape(-1)).astype(int)
        
        self.preserve_out_c = copy.deepcopy(self.out_channels)
        self.preserve_out_c = (self.preserve_out_c*np.array(self.preserve_ratio_p).reshape(-1)).astype(int)
        self.pruned_out_c = self.out_channels - self.preserve_out_c


    def model_to_state(self, flops_list, ori_flops_list, actions_p):
        try:
            current_state = construct_state(self.prune_quant_net, self.model_name, self.preserve_out_c, 
                                               self.preserve_in_c, flops_list, ori_flops_list, actions_p)
        except:
            # First iteration requires a dummy model (coz it has not been quantized yet). This is called in def reset()
            from networks.cnn1d import CNN1D_TrafficClassification
            from pytorch_quantization import quant_modules
            quant_modules.initialize()
            model = CNN1D_TrafficClassification(input_ch=self.input_x.shape[1], num_classes=16).to(self.device)
            current_state = construct_state(model, self.model_name, self.preserve_out_c, 
                                               self.preserve_in_c, flops_list, ori_flops_list, actions_p)
        
        current_state = current_state.cuda()
        return current_state
