import os
import sys
import time
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tqdm import tqdm
from pytorch_quantization import quant_modules
from networks.cnn1d import CNN1D_TrafficClassification
from utils.training_utils import create_data_loaders
from prune_quant.quant_utils import evaluate, save_model_state_dict, load_fp32_model, benchmark_against_NiN, benchmark_without_NiN
from prune_quant.prune import prune_permenantly
from prune_quant.quant_convert import convert_to_onnx


def finetune(net, train_loader, val_loader, dataset_name, model_name, epochs, lr, qat_momentum, qat_wd, current_val_loss, preserve_array_prune, bitwidths, device):

    net.to(device)
    qat_optimizer = optim.SGD(net.parameters(), lr, qat_momentum, qat_wd)
    qat_criterion = nn.CrossEntropyLoss().to(device)   # training settings

    for epoch in range(0, epochs):

        net.train()
        epoch_loss = 0
        epoch_acc = 0
        train_running_loss = 0.0
        train_running_correct = 0
        counter = 0

        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):

            counter += 1
            rawpacket, labels = data
            rawpacket = rawpacket.to(device)
            labels = labels.to(device)
            qat_optimizer.zero_grad()

            # Forward pass.
            outputs = net(rawpacket)

            # Calculate the loss.
            loss = qat_criterion(outputs, labels)
            train_running_loss += loss.item()

            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)   # highest probability along 2nd dimension [1] which is the class
            train_running_correct += (preds == labels).sum().item()

            # Backpropagation
            loss.backward()

            # Update the weights.
            qat_optimizer.step()
        
        # Loss and accuracy for the complete epoch.
        epoch_loss = train_running_loss / counter
        epoch_acc = 100. * (train_running_correct / len(train_loader.dataset))

        print(f"Epoch {epoch}: Training loss: {epoch_loss:.3f}, training acc: {epoch_acc:.3f}")
        val_loss, val_acc, _ = evaluate(net, val_loader, qat_criterion, device)
        print(f"Epoch {epoch}: Validation loss: {val_loss:.3f}, validation acc: {val_acc:.3f}")

        # Save the model
        if val_loss < current_val_loss:
            current_val_loss = val_loss
            print(f"Saving best model with {val_acc}% validation acc")
            save_model_state_dict(net, model_name, dataset_name, preserve_array_prune, bitwidths)




if __name__ == "__main__":

    # Set hyperparameters
    start_time  = time.time()
    device      = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    n_worker    = 0
    parser      = argparse.ArgumentParser()
    criterion   = nn.CrossEntropyLoss()
    parser.add_argument('--epochs', default=120, help='train with how many epochs')
    parser.add_argument('--batch_size', default=1, help='desired batch size to deploy the compressed model')
    parser.add_argument('--dataset', default='iscx2016vpn', help='which dataset to train on')
    parser.add_argument('--lr', default=0.001, help='learning rate')
    parser.add_argument('--qat_momentum', default=0.9, type=float, help='momentum for quantize aware training')
    parser.add_argument('--qat_wd', default=4e-5, type=float, help='weight decay for quantize aware training')
    parser.add_argument('--eval_trt', default='false', help='only set this to \'true\' after converting onnx to trt engine')
    parser.add_argument('--trt_mem_cpu_breakdown', default='false', help='Set to \'true\' to get the timing breakdown of memory vs computation')
    args = parser.parse_args()
    inf_batch_size = int(args.batch_size)
    if args.dataset == "iscx2016vpn":
        dataset     = os.path.join('data', 'datasets', 'iscx2016vpn-pytorch')
        train_ratio = 0.65
        val_ratio   = 0.15
    elif args.dataset == "ustctfc2016":
        dataset     = os.path.join('data', 'datasets', 'ustc-tfc2016-pytorch')
        train_ratio = 0.8
        val_ratio   = 0.1
    elif args.dataset == "ciciot2022":
        dataset     = os.path.join('data', 'datasets', 'ciciot2022-pytorch')
        train_ratio = 0.65
        val_ratio   = 0.15
    elif args.dataset == "itcnetaudio5":
        dataset     = os.path.join('data', 'datasets', 'itcnetaudio5-pytorch')
        train_ratio = 0.65
        val_ratio   = 0.15
    train_loader, valid_loader, test_loader, classes = create_data_loaders(dataset, inf_batch_size, n_worker, train_ratio, val_ratio)
    example_inputs = (next(iter(test_loader))[0]).to(device)



    # Check whether to benchmark the trt model or to finetune
    if args.eval_trt == "true":
        if args.dataset == "iscx2016vpn":
            trt_engine_path = os.path.join("networks","quantized_models","iscx2016vpn","model_engine.trt")
            path_own_model  = os.path.join('networks', 'pretrained_models', 'iscx2016vpn', 'CNN1D_TrafficClassification_best_model_without_aux.pth')
            path_NiN_model  = os.path.join('networks', 'pretrained_models', 'iscx2016vpn', 'NiN_CNN1D_TrafficClassification_best_model_without_aux.pth')
            net             = load_fp32_model(path=path_own_model, input_ch=example_inputs.shape[1], num_classes=classes, device=device)
            net_NiN         = load_fp32_model(path=path_NiN_model, input_ch=example_inputs.shape[1], num_classes=classes, device=device)
            benchmark_against_NiN(net, net_NiN, trt_engine_path, test_loader, inf_batch_size, classes, criterion, device)
            from prune_quant.quant_utils import save_prune_quant_model_to_check_size
            save_prune_quant_model_to_check_size()
            if args.trt_mem_cpu_breakdown == "true":
                from prune_quant.quant_utils import write_mem_comp_breakdown_prune_quant, write_mem_comp_breakdown_baseline
                inf_batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
                for inf_batch_size in inf_batch_sizes:
                    _, _, test_loader, classes = create_data_loaders(dataset, inf_batch_size, n_worker, train_ratio, val_ratio)
                    write_mem_comp_breakdown_baseline(net, net_NiN, trt_engine_path, test_loader, inf_batch_size, classes, criterion, device)
                    write_mem_comp_breakdown_prune_quant(net, net_NiN, trt_engine_path, test_loader, inf_batch_size, classes, criterion, device)

            exit()
        elif args.dataset == "ustctfc2016":
            trt_engine_path = os.path.join("networks","quantized_models","ustc-tfc2016","model_engine.trt")
            path_own_model  = os.path.join('networks', 'pretrained_models', 'ustc-tfc2016', 'CNN1D_TrafficClassification_best_model_without_aux.pth')
            path_NiN_model  = os.path.join('networks', 'pretrained_models', 'ustc-tfc2016', 'NiN_CNN1D_TrafficClassification_best_model_without_aux.pth')
            net             = load_fp32_model(path=path_own_model, input_ch=example_inputs.shape[1], num_classes=classes, device=device)
            net_NiN         = load_fp32_model(path=path_NiN_model, input_ch=example_inputs.shape[1], num_classes=classes, device=device)
            benchmark_against_NiN(net, net_NiN, trt_engine_path, test_loader, inf_batch_size, classes, criterion, device)
            exit()
        
        elif args.dataset == "ciciot2022":
            trt_engine_path = os.path.join("networks","quantized_models","ciciot2022","model_engine.trt")
            path_own_model  = os.path.join('networks', 'pretrained_models', 'ciciot2022', 'CNN1D_TrafficClassification_best_model_without_aux.pth')
            net             = load_fp32_model(path=path_own_model, input_ch=example_inputs.shape[1], num_classes=classes, device=device)
            benchmark_without_NiN(net, trt_engine_path, test_loader, inf_batch_size, classes, criterion, device)
            exit()
        
        elif args.dataset == "itcnetaudio5":
            trt_engine_path = os.path.join("networks","quantized_models","itcnetaudio5","model_engine.trt")
            path_own_model  = os.path.join('networks', 'pretrained_models', 'itcnetaudio5', 'CNN1D_TrafficClassification_best_model_without_aux.pth')
            net             = load_fp32_model(path=path_own_model, input_ch=example_inputs.shape[1], num_classes=classes, device=device)
            benchmark_without_NiN(net, trt_engine_path, test_loader, inf_batch_size, classes, criterion, device)
            exit()



    # Load prune and quantized model
    quant_modules.initialize()
    net          = CNN1D_TrafficClassification(input_ch=example_inputs.shape[1], num_classes=classes).to(device)
    model_name   = net.__class__.__name__

    if args.dataset == "iscx2016vpn":
        checkpoint   = torch.load(os.path.join("networks","quantized_models","iscx2016vpn", f"{model_name}_{args.dataset}.pt"), map_location=device)
    elif args.dataset == "ustctfc2016":
        checkpoint   = torch.load(os.path.join("networks","quantized_models","ustc-tfc2016", f"{model_name}_{args.dataset}.pt"), map_location=device)
    elif args.dataset == "ciciot2022":
        checkpoint   = torch.load(os.path.join("networks","quantized_models","ciciot2022", f"{model_name}_{args.dataset}.pt"), map_location=device)
    elif args.dataset == "itcnetaudio5":
        checkpoint   = torch.load(os.path.join("networks","quantized_models","itcnetaudio5", f"{model_name}_{args.dataset}.pt"), map_location=device)

    preserve_array_prune = checkpoint['preserve_array_prune']
    bitwidths = checkpoint['bitwidths']

    for name, module in net.named_modules():
        if isinstance(module, torch.nn.Conv1d): # dummy pruning to add masks
            prune.ln_structured(module, name='weight', amount=0., n=2, dim=0)   

    sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    net.load_state_dict(sd, strict=True)



    # Get model's accuracy with targeted inference batch_size
    best_acc = 0
    current_loss, current_acc, _ = evaluate(net, valid_loader, criterion, device)
    print(f"Current accuracy is: {current_acc:.2f}%")



    # Finetune model
    finetune(net, train_loader, valid_loader, str(args.dataset), model_name,
             int(args.epochs), float(args.lr), float(args.qat_momentum), float(args.qat_wd), 
             current_loss, preserve_array_prune, bitwidths, device)


    # Save as onnx
    prune_permenantly(net)
    if args.dataset == "iscx2016vpn":
        onnx_path = os.path.join("networks","quantized_models","iscx2016vpn","model.onnx")
    elif args.dataset == "ustctfc2016":
        onnx_path = os.path.join("networks","quantized_models","ustc-tfc2016","model.onnx")
    elif args.dataset == "ciciot2022":
        onnx_path = os.path.join("networks","quantized_models","ciciot2022","model.onnx")
    elif args.dataset == "itcnetaudio5":
        onnx_path = os.path.join("networks","quantized_models","itcnetaudio5","model.onnx")
    convert_to_onnx(net, example_inputs, onnx_path)
    print(f"Export model as onnx at {onnx_path}")
    print("Please run trtexec to convert to .trt engine")

