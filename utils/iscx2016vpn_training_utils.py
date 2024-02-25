import os
import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader, Subset, random_split
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from networks.cnn1d import CNN1D_TrafficClassification_with_auxiliary, CNN1D_TrafficClassification
from networks.cnn1d_NiN import NiN_CNN1D_TrafficClassification_with_auxiliary, NiN_CNN1D_TrafficClassification  #using other work's model that uses NiN


# Training function.
def train(model, train_loader, optimizer, criterion, device):
    """
    Basic train function.
    Basically iterate through the train dataset (in the form of <DataLoader>), 
    Infer, calc loss, calc acc, BP, updt weight.
    return epoch_loss, epoch_acc
    """
    model.train()
    print('Training evaluation')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    
    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        counter += 1
        rawpacket, labels = data
        rawpacket = rawpacket.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        # Forward pass.
        outputs = model(rawpacket)

        # Calculate the loss.
        if isinstance(outputs, tuple):  # Check if model outputs auxiliary predictions.
            main_output, auxiliary_outputs = outputs
            # Main loss
            loss = criterion(main_output, labels)
            # Auxiliary losses
            for aux_output in auxiliary_outputs:
                aux_loss = criterion(aux_output, labels)
                loss += aux_loss  # Summing main loss with auxiliary losses
        else:
            main_output = outputs
            loss = criterion(main_output, labels)

        train_running_loss += loss.item()
        # Calculate the accuracy. (Only from main_outputs, not from auxiliary)
        _, preds = torch.max(main_output.data, 1)
        train_running_correct += (preds == labels).sum().item()

        # Backpropagation
        loss.backward()

        # Update the weights.
        optimizer.step()
    
    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    # epoch_acc = 100. * (train_running_correct / len(train_loader.dataset))
    epoch_acc = 100. * (train_running_correct / len(train_loader.dataset))
    return epoch_loss, epoch_acc



# Validation function.
def validate(model, dataloader, criterion, device):
    """
    Basic validate function.
    Basically iterate through the valdiation/test dataset (in the form of <DataLoader>), 
    Infer, calc loss, calc acc.
    return epoch_loss, epoch_acc
    """
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            counter += 1
            
            rawpacket, labels = data
            rawpacket = rawpacket.to(device)
            labels = labels.to(device)

            # Forward pass.
            outputs = model(rawpacket)
            
            # Ensure compatibility if model returns auxiliary outputs
            main_output = outputs if not isinstance(outputs, tuple) else outputs[0]

            # Calculate the loss.
            loss = criterion(main_output, labels)
            valid_running_loss += loss.item()

            # Calculate the accuracy.
            _, preds = torch.max(main_output.data, 1)
            valid_running_correct += (preds == labels).sum().item()
        
    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(dataloader.dataset))
    return epoch_loss, epoch_acc



# Prepare an object that store best model at a given epoch
class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_loss=float('inf')  #lost initialize as infinity
    ):
        self.best_valid_loss = best_valid_loss
        
    def __call__(                   #whenever this class is called, need to pass in this parameter
        self, current_valid_loss, 
        epoch, model, optimizer, criterion
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, os.path.join('networks', 'pretrained_models', 'iscx2016vpn', 'best_model.pth'))



def save_plots(train_acc, valid_acc, train_loss, valid_loss, name=None):
    """
    Function to save the loss and accuracy plots to disk.
    The train_acc, valid_acc, train_loss, and valid_loss are lists containing the respective values for each epoch.
    """
    # Accuracy plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='tab:blue', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='tab:red', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join('networks', 'pretrained_models', 'iscx2016vpn', name+'_accuracy.png'))
    
    # Loss plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='tab:blue', linestyle='-', 
        label='train loss (including auxiliary loss)'
    )
    plt.plot(
        valid_loss, color='tab:red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join('networks', 'pretrained_models', 'iscx2016vpn', name+'_loss.png'))



class ISCX2016VPNPacketDataset(Dataset):
    def __init__(self, directory):
        """
        Args:
            directory (str): Directory with all .pt files, each named after its class.
        """
        self.filepaths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pt')]
        self.labels = {os.path.splitext(os.path.basename(f))[0]: i for i, f in enumerate(self.filepaths)}
        self.data = []
        self.load_data()

    def load_data(self):
        for filepath in self.filepaths:
            class_name = os.path.splitext(os.path.basename(filepath))[0]
            label = self.labels[class_name]
            packet_data = torch.load(filepath)
            
            no_of_packet = 0
            for packet in packet_data:
                no_of_packet += 1
                self.data.append((packet, label))
            print(f"class name: {class_name} \t label: {label} \t no. packet: {no_of_packet}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        packet, label = self.data[idx]
        # Add a channel dimension to packet data
        packet = packet.unsqueeze(0)  # This transforms the shape from [1500] to [1, 1500]
        return packet, label
    


def create_data_loaders_iscx2016vpn(data_directory, batch_size, n_worker, train_ratio, val_ratio):
    # Initialize the dataset
    full_dataset = ISCX2016VPNPacketDataset(data_directory)

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

    return train_loader, val_loader, test_loader, len(full_dataset.labels)



if __name__ == '__main__':

    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    save_best_model = SaveBestModel() # Create an object that store the best model at a given epoch

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=120, help='train with how many epochs')
    parser.add_argument('--batch_size', default=1024, help='batch size for training pretrained model')
    parser.add_argument('--dataset', default='iscx2016vpn', help='which dataset to train on')
    parser.add_argument('--NiN_model', default='false', help='string argument to train the nin model instead. give \'true\' to use the NiN model, or else dont put this argument')
    parser.add_argument('--lr', default=0.001, help='learning rate')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_ratio = 0.65
    val_ratio   = 0.15
    test_ratio  = 1.0 - train_ratio - val_ratio
    train_loader, valid_loader, test_loader, classes = create_data_loaders_iscx2016vpn(
        os.path.join('data', 'datasets', 'iscx2016vpn-pytorch'),
        batch_size=1024,
        n_worker=0,
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )

    
    if 'true' in args.NiN_model:
        print("Training the NiN model")
        model = NiN_CNN1D_TrafficClassification_with_auxiliary(input_ch=1, num_classes=classes).to(device)
    else:
        model = CNN1D_TrafficClassification(input_ch=1, num_classes=classes).to(device)

    # load best model if it exist / checkpoint
    prev_path = Path(os.path.join('networks', 'pretrained_models', 'iscx2016vpn', 'best_model.pth'))
    if (prev_path.is_file() == True):
        print(f"Previous trained model exist at {prev_path}\n")
        print("Loading the model...")
        model.load_state_dict(torch.load(prev_path), strict=False)

    # hyperparameter for training
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Start the training.
    for epoch in range(int(args.epochs)):
        print(f"[INFO]: Epoch {epoch+1} of {args.epochs}")
        train_epoch_loss, train_epoch_acc = train(model, train_loader, optimizer, criterion, device)
        valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader, criterion, device)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        print('-'*50)

        save_best_model(valid_epoch_loss, epoch, model, optimizer, criterion) # save the best model in this current epoch

    # Save the loss and accuracy plots.
    save_plots(train_acc, valid_acc, train_loss, valid_loss, name='CNN1D_Model')
    print(f"TRAINING COMPLETE")
    print(f"Training ratio :{train_ratio}")
    print(f"Val ratio      :{val_ratio}")
    print(f"Test ratio     :{test_ratio}")
    
    # Test accuracy (of unseen test_data) on the best model, using the inference model (without auxiliary)
    if 'true' in args.NiN_model:
        inf_model = NiN_CNN1D_TrafficClassification(input_ch=1, num_classes=classes).to(device)
    else:
        inf_model = CNN1D_TrafficClassification(input_ch=1, num_classes=classes).to(device)

    checkpoint = torch.load(prev_path, map_location=device)
    sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    inf_model.load_state_dict(sd, strict=False)
    test_loss, test_acc = validate(inf_model, test_loader, criterion, device)
    torch.save(inf_model.state_dict(), os.path.join('networks', 'pretrained_models', 'iscx2016vpn', 'best_model_without_aux.pth'))
    
    print(f"Acc on unseen test data (not validation dataset): {test_acc}")
