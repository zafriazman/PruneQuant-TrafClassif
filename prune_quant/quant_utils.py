import os
import time
import torch
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
from networks.cnn1d import CNN1D_TrafficClassification
from networks.cnn1d_NiN import NiN_CNN1D_TrafficClassification


def evaluate_trt(engine_path, dataloader, batch_size, num_classes):
    
    def predict(batch, current_batch_size, num_classes): # result gets copied into output
        output = np.empty([current_batch_size, num_classes], dtype=np.float32)  # Adjusted output allocation
        # transfer input data to device
        cuda.memcpy_htod_async(d_input, batch, stream)
        # execute model
        context.execute_async_v2(bindings, stream.handle, None)
        # transfer predictions back
        cuda.memcpy_dtoh_async(output, d_output, stream)
        # syncronize threads
        stream.synchronize()
        return output
    

    with open(engine_path, 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime, runtime.deserialize_cuda_engine(f.read()) as engine, engine.create_execution_context() as context:
        total = 0
        correct = 0
        avg_time = None
        time_per_batch = []
        
        # Warmup
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        eg_inputs, eg_labels = (next(iter(dataloader)))
        eg_inputs = eg_inputs.numpy()
        eg_labels = eg_labels.numpy()
        bsize_allocate = eg_inputs.shape[0]
        d_input  = cuda.mem_alloc(1 * eg_inputs.nbytes)
        d_output = cuda.mem_alloc(int(1 * np.prod([bsize_allocate, num_classes]) * np.float32().itemsize))
        bindings = [int(d_input), int(d_output)]
        stream   = cuda.Stream()
        eg_pred  = predict(eg_inputs, bsize_allocate, num_classes)

        for rawpackets, labels in dataloader:
            
            input_batch = rawpackets.numpy()
            labels = labels.numpy()
            current_batch_size = input_batch.shape[0]

            # Allocate input and output memory, give TRT pointers (bindings) to it:
            d_input = cuda.mem_alloc(1 * input_batch.nbytes)
            d_output = cuda.mem_alloc(int(1 * np.prod([current_batch_size, num_classes]) * np.float32().itemsize))
            bindings = [int(d_input), int(d_output)]

            stream = cuda.Stream()
            start_time = time.time()
            preds = predict(input_batch, current_batch_size, num_classes)
            time_per_batch.append(time.time() - start_time)
            pred_labels = []
            for pred in preds:
                pred_label = (-pred).argsort()[0]
                pred_labels.append(pred_label)

            total += len(labels)
            correct += (pred_labels == labels).sum()
            
        
        avg_time = sum(time_per_batch) / len(time_per_batch)
    
    return correct/total*100., avg_time


def evaluate(model, dataloader, criterion, device):

    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    avg_time = None
    time_per_batch = []
    
    # warm up
    eg_inputs, eg_labels = (next(iter(dataloader)))
    eg_inputs = eg_inputs.to(device)
    eg_labels = eg_labels.to(device)
    eg_out    = model(eg_inputs)

    with torch.no_grad():
        for i, (rawpacket, labels) in enumerate(dataloader):
            counter += 1
            
            # for fairness with TRT, also include time to load from RAM to gpu mem
            start_time = time.time()
            rawpacket = rawpacket.to(device)
            labels = labels.to(device)

            # Forward pass.
            outputs = model(rawpacket)
            time_per_batch.append(time.time() - start_time)
            
            # Ensure compatibility if model returns auxiliary outputs
            main_output = outputs if not isinstance(outputs, tuple) else outputs[0]

            # Calculate the loss.
            loss = criterion(main_output, labels)
            valid_running_loss += loss.item()

            # Calculate the accuracy.
            _, preds = torch.max(main_output.data, 1)
            valid_running_correct += (preds == labels).sum().item()
    
    # Calculate averagte time per batch
    avg_time = sum(time_per_batch) / len(time_per_batch)
        
    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(dataloader.dataset))
    return epoch_loss, epoch_acc, avg_time




def print_size_of_model(model):
    if isinstance(model, torch.jit.RecursiveScriptModule):
        torch.jit.save(model, "temp.p")
    else:
        torch.jit.save(torch.jit.script(model), "temp.p")
    print("Size (kB):", os.path.getsize("temp.p")/1e3)
    os.remove("temp.p")



def save_model_state_dict(model, action_prune, action_quant):
    content = {
        'state_dict'  : model.state_dict(),
        'action_prune': action_prune,
        'action_quant': action_quant
    }
    checkpoint = os.path.join("networks","quantized_models","iscx2016vpn", "model.pt")
    torch.save(content, checkpoint)



def load_fp32_model(path, input_ch, num_classes, device):

    if "NiN" in os.path.basename(path):
        net = NiN_CNN1D_TrafficClassification(input_ch=input_ch, num_classes=num_classes).to(device)
    else:
        net = CNN1D_TrafficClassification(input_ch=input_ch, num_classes=num_classes).to(device)
        
    checkpoint = torch.load(path, map_location=device)
    sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    net.load_state_dict(sd, strict=True)
    return net



def benchmark_against_NiN(net, net_NiN, trt_engine_path, test_loader, batch_size, classes, criterion, device):
    
    print("Benchmarking...")

    # NiN model
    start_time_inf_fp32_NiN        = time.time()
    _, test_acc_NiN, time_fp32_NiN = evaluate(net_NiN, test_loader, criterion, device)
    fp32_inf_time_NiN              = time.time() - start_time_inf_fp32_NiN

    # My model (baseline on GPU)
    start_time_inf_fp32            = time.time()
    _, test_acc_fp32, time_fp32    = evaluate(net, test_loader, criterion, device)
    fp32_inf_time                  = time.time() - start_time_inf_fp32

    # My model (baseline on CPU)
    start_time_inf_cpu             = time.time()
    _, test_acc_cpu, time_cpu    = evaluate(net.cpu(), test_loader, criterion, torch.device("cpu"))
    cpu_inf_time                  = time.time() - start_time_inf_fp32
    
    # prune and quantized model
    start_time_inf               = time.time()
    test_acc, avg_time_per_batch = evaluate_trt(trt_engine_path, test_loader, batch_size, classes)
    trt_inf_time                 = time.time() - start_time_inf


    print(f"\n")
    print(f"Total speedup from GPU FP32 to TensorRT INT8 is {(fp32_inf_time/trt_inf_time):.4f}x")
    print(f"Speedup from GPU FP32 to TensorRT INT8 for each batch is {(time_fp32/avg_time_per_batch):.4f}x")

    print(f"Using CUDA (TensorRT)")
    print(f"Acc on unseen test data (not validation dataset):           {test_acc:.4f}")
    print(f"Using CUDA (TensorRT): Elapsed time for 20% of dataset:     {trt_inf_time:.4f} seconds")
    print(f"Using CUDA (TensorRT): Average time per batch:              {avg_time_per_batch:4f} seconds")
    
    print(f"\n")
    print(f"Using GPU FP32 (my model)")
    print(f"Acc on unseen test data (not validation dataset):           {test_acc_fp32}")
    print(f"Using GPU FP32 (my model): Elapsed time for 20% of dataset: {fp32_inf_time:.4f} seconds")
    print(f"Using GPU FP32 (my model): Average time per batch:          {time_fp32:4f} seconds")

    print(f"\n")
    print(f"Using CPU FP32 (my model)")
    print(f"Acc on unseen test data (not validation dataset):           {test_acc_cpu}")
    print(f"Using CPU FP32 (my model): Elapsed time for 20% of dataset: {cpu_inf_time:.4f} seconds")
    print(f"Using CPU FP32 (my model): Average time per batch:          {time_cpu:4f} seconds")

    print(f"\n")
    print(f"Using GPU FP32 (NiN model)")
    print(f"Acc on unseen test data (not validation dataset):           {test_acc_NiN}")
    print(f"Using GPU FP32 (NiN model): Elapsed time for 20% of dataset:{fp32_inf_time_NiN:.4f} seconds")
    print(f"Using GPU FP32 (NiN model): Average time per batch:         {time_fp32_NiN:4f} seconds")


def evaluate_trt_backup_orig(engine_path, dataloader, batch_size, num_classes):
    
    def predict(batch, current_batch_size, num_classes): # result gets copied into output
        output = np.empty([current_batch_size, num_classes], dtype=np.float32)  # Adjusted output allocation
        # transfer input data to device
        cuda.memcpy_htod_async(d_input, batch, stream)
        # execute model
        context.execute_async_v2(bindings, stream.handle, None)
        # transfer predictions back
        cuda.memcpy_dtoh_async(output, d_output, stream)
        # syncronize threads
        stream.synchronize()
        return output
    
    with open(engine_path, 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime, runtime.deserialize_cuda_engine(f.read()) as engine, engine.create_execution_context() as context:
        total = 0
        correct = 0
        avg_time = None
        time_per_batch = []
        for rawpackets, labels in dataloader:
            start_time = time.time()
            input_batch = rawpackets.numpy()
            labels = labels.numpy()
            current_batch_size = input_batch.shape[0]

            # Now allocate input and output memory, give TRT pointers (bindings) to it:
            d_input = cuda.mem_alloc(1 * input_batch.nbytes)
            d_output = cuda.mem_alloc(int(1 * np.prod([current_batch_size, num_classes]) * np.float32().itemsize))
            bindings = [int(d_input), int(d_output)]

            stream = cuda.Stream()
            preds = predict(input_batch, current_batch_size, num_classes)
            pred_labels = []
            for pred in preds:
                pred_label = (-pred).argsort()[0]
                pred_labels.append(pred_label)

            total += len(labels)
            correct += (pred_labels == labels).sum()
            time_per_batch.append(time.time() - start_time)
        
        avg_time = sum(time_per_batch) / len(time_per_batch)
    
    return correct/total, avg_time