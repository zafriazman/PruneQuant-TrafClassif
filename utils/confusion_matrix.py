import os
import sys
import time
import torch
import numpy as np
import torch.nn as nn
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import seaborn as sns
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from torch.utils.data import Dataset, DataLoader, random_split
from utils.training_utils import PacketDataset
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt



def get_true_and_pred_labels(engine_path, dataloader, batch_size, num_classes):

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


    true_labels = []
    predicted_labels = []

    with open(engine_path, 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime, runtime.deserialize_cuda_engine(f.read()) as engine, engine.create_execution_context() as context:

        for rawpackets, labels in dataloader:
            
            input_batch = rawpackets.numpy()
            labels = labels.numpy()
            current_batch_size = input_batch.shape[0]

            # Allocate input and output memory, give TRT pointers (bindings) to it:
            d_input = cuda.mem_alloc(1 * input_batch.nbytes)
            d_output = cuda.mem_alloc(int(1 * np.prod([current_batch_size, num_classes]) * np.float32().itemsize))
            bindings = [int(d_input), int(d_output)]

            stream = cuda.Stream()
            preds = predict(input_batch, current_batch_size, num_classes)

            for i, pred in enumerate(preds):
                pred_label = (-pred).argsort()[0]
                predicted_labels.append(pred_label)
                true_labels.append(labels[i])


    return true_labels, predicted_labels




def create_data_loaders_with_labels(data_directory, batch_size, n_worker, train_ratio, val_ratio):
    # Initialize the dataset
    full_dataset = PacketDataset(data_directory)

    class_labels = full_dataset.labels
    class_labels_list = []
    for index, (key, value) in enumerate(class_labels.items()):
        if index == value:
            class_labels_list.append(key)
        else:
            raise IndexError(f"They key is not enumerated in sequential order (eg: suppose to be 1, 2, 3 ...) but got (1, 3, 2, ...)")

    # Calculate the sizes for training and validation splits
    train_size = int(len(full_dataset) * train_ratio)
    val_size = int(len(full_dataset) * val_ratio)
    test_size = len(full_dataset) - train_size - val_size

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    # Create DataLoader for training and validation datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_worker)    # Still shuffle evnthough the 7000 data is already randomized
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_worker)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=n_worker)

    return train_loader, val_loader, test_loader, len(full_dataset.labels), class_labels_list



start_time  = time.time()
device      = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
n_worker    = 0
criterion   = nn.CrossEntropyLoss()
dataset     = os.path.join('data', 'datasets', 'iscx2016vpn-pytorch')
train_ratio = 0.65
val_ratio   = 0.15
bsize       = 32

train_loader, valid_loader, test_loader, n_classes, class_labels = create_data_loaders_with_labels(dataset, bsize, n_worker, train_ratio, val_ratio)

trt_engine_path = os.path.join("networks","quantized_models","iscx2016vpn","model_engine.trt")

true_labels, pred_labels = get_true_and_pred_labels(trt_engine_path, test_loader, bsize, n_classes)


class_labels = tuple(class_labels)

# Convert lists to numpy arrays
true_labels = np.array(true_labels)
pred_labels = np.array(pred_labels)
conf_matrix = confusion_matrix(true_labels, pred_labels)

# Build confusion matrix
cf_matrix = confusion_matrix(true_labels, pred_labels)
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in class_labels],
                     columns = [i for i in class_labels])
plt.figure(figsize = (12,12))
plt.rc('font', family='serif', serif='Times new Roman')
sns.heatmap(df_cm, annot=True, fmt='.2f', cmap='YlOrBr')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.yticks(rotation=0)
plt.show()
#plt.savefig('output.png')

