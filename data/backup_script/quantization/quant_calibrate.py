import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def calibrate(quantized_net, data_loader, epochs, device):

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
    