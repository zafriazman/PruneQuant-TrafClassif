import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_quantization.nn as quant_nn
import pytorch_quantization.calib as calib

from tqdm import tqdm


def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)

                # print(F"{name:40}: {module}\n")
    model.cuda()



def collect_stats(model, data_loader, num_batches):
    """Feed data to the network and collect statistics"""
    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()  # calib in FP32
                module.enable_calib()
            else:
                module.disable()

    # Feed data to the network for collecting stats
    for i, (rawpacket, _) in tqdm(enumerate(data_loader), total=num_batches):
        model(rawpacket.cuda())
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()

def qat_train(quantized_net, data_loader, epochs, device):

    quantized_net.to(device)
    qat_optimizer = optim.SGD(quantized_net.parameters(), lr=0.01)
    qat_criterion = nn.CrossEntropyLoss().to(device)   # training settings

    """ for rawpacket, label in data_loader:
        rawpacket = rawpacket.to(device)
        label = label.to(device)
        quantized_net(rawpacket) """

    for epoch in range(0, epochs):

        quantized_net.train()
        epoch_loss = 0
        epoch_acc = 0
        train_running_loss = 0.0
        train_running_correct = 0
        counter = 0

        for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):

            counter += 1
            rawpacket, labels = data
            rawpacket = rawpacket.to(device)
            labels = labels.to(device)
            qat_optimizer.zero_grad()

            # Forward pass.
            outputs = quantized_net(rawpacket)

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
        epoch_acc = 100. * (train_running_correct / len(data_loader.dataset))

        print(f"Epoch {epoch}: Training loss: {epoch_loss:.3f}, training acc: {epoch_acc:.3f}")
    