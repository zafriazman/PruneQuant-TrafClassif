import os
import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

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
    Basically iterate through the train dataset (in the form of <DataLoader>), 
    Infer, calc loss, calc acc, BP (with self distillation), updt weight.
    return epoch_loss, epoch_acc
    """
    model.train()
    print('Training evaluation')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0

    if "NiN" in model.__class__.__name__:
        alpha = [0.1, 0.1, 0.8]  # Weightage for different submodule losses
        beta  = 0.05             # Weightage for KL divergence loss
        gamma = 0.000001         # Weightage for L2 norm feature loss
    else:
        alpha = [1.0, 1.0, 1.0, 1.0]
        beta  = 0.000
        gamma = 0.000000
    
    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        counter += 1
        rawpacket, labels = data
        rawpacket = rawpacket.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        # Forward pass.
        outputs, outputs_fmaps = model(rawpacket)

        loss = 0
        final_output = outputs[-1]
        final_features = outputs_fmaps[-1]

        loss += criterion(final_output, labels)
        temperature = 10.0    # following temperature from "Two-stage distillation... NiN paper"

        # CrossEntropy loss for each submodule output (excluding final output i.e. real classification)
        for i, output in enumerate(outputs[:-1]):
            loss += alpha[i] * criterion(output, labels)
        
        # KL divergernce loss between first 2 submodule classification vs deepest classification
        # Exclude last two, following "Two-stage distillation... NiN paper"
        for i, output in enumerate(outputs[:-2]):
            loss += beta * KL_DivergenceLoss(output, final_output, temperature)
                
        # L2 Norm loss - distance between feature maps
        # Exclude last two, following "Two-stage distillation... NiN paper"
        for i, feature in enumerate(outputs_fmaps[:-2]):
            target_feature_size = final_features.shape[2]
            adaptive_pool = nn.AdaptiveMaxPool1d(target_feature_size)
            adjusted_feature = adaptive_pool(feature)
            l2_norm = torch.norm(adjusted_feature - final_features, p=2).item()
            loss += gamma * l2_norm

        train_running_loss += loss.item()
        # Calculate the accuracy. (Only from final_output, not from auxiliary)
        _, preds = torch.max(final_output.data, 1)
        train_running_correct += (preds == labels).sum().item()

        # Backpropagation
        loss.backward()

        # Update the weights.
        optimizer.step()
    
    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(train_loader.dataset))
    return epoch_loss, epoch_acc



# Self distillation's loss
def CrossEntropyLoss(outputs, targets):
    # Temperature = 1, i.e., hard predict
    criterion = nn.CrossEntropyLoss()
    loss_CE = criterion(outputs, targets)
    return loss_CE

def KL_DivergenceLoss(outputs, targets, temperature):
    log_softmax_outputs = F.log_softmax(outputs/temperature, dim=1)
    softmax_targets = F.softmax(targets/temperature, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()



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
            criterion = nn.CrossEntropyLoss()
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
        epoch, model, optimizer, criterion, dataset
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            model_name = model.__class__.__name__
            if dataset == "iscx2016vpn":
                torch.save({
                    'epoch': epoch+1,
                    'state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': criterion,
                    }, os.path.join('networks', 'pretrained_models', 'iscx2016vpn', (model_name+'_best_model.pth')))
            elif dataset == "ustctfc2016":
                torch.save({
                'epoch': epoch+1,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, os.path.join('networks', 'pretrained_models', 'ustc-tfc2016', (model_name+'_best_model.pth')))



def save_plots(train_acc, valid_acc, train_loss, valid_loss, name, dataset):
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
    if dataset == "iscx2016vpn":
        plt.savefig(os.path.join('networks', 'pretrained_models', 'iscx2016vpn', name+'_accuracy.png'))
    elif dataset == "ustctfc2016":
        plt.savefig(os.path.join('networks', 'pretrained_models', 'ustc-tfc2016', name+'_accuracy.png'))
    
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
    if dataset == "iscx2016vpn":
        plt.savefig(os.path.join('networks', 'pretrained_models', 'iscx2016vpn', name+'_loss.png'))
    elif dataset == "ustctfc2016":
        plt.savefig(os.path.join('networks', 'pretrained_models', 'ustc-tfc2016', name+'_loss.png'))



class PacketDataset(Dataset):
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
    


def create_data_loaders(data_directory, batch_size, n_worker, train_ratio, val_ratio):
    # Initialize the dataset
    full_dataset = PacketDataset(data_directory)

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

    

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=120, help='train with how many epochs')
    parser.add_argument('--batch_size', default=1024, help='batch size for training pretrained model')
    parser.add_argument('--dataset', default='iscx2016vpn', help='which dataset to train on')
    parser.add_argument('--NiN_model', default='false', help='string argument to train the nin model instead. give \'true\' to use the NiN model, or else dont put this argument')
    parser.add_argument('--lr', default=0.001, help='learning rate')
    args = parser.parse_args()

    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    save_best_model = SaveBestModel() # Create an object that store the best model at a given epoch

    if args.dataset == "iscx2016vpn":
        train_ratio = 0.65
        val_ratio   = 0.15
        test_ratio  = 1.0 - train_ratio - val_ratio
        train_loader, valid_loader, test_loader, classes = create_data_loaders(
            os.path.join('data', 'datasets', 'iscx2016vpn-pytorch'),
            batch_size=int(args.batch_size),
            n_worker=0,
            train_ratio=train_ratio,
            val_ratio=val_ratio
        )
    elif args.dataset == "ustctfc2016":
        train_ratio = 0.8
        val_ratio   = 0.1
        test_ratio  = 1.0 - train_ratio - val_ratio
        train_loader, valid_loader, test_loader, classes = create_data_loaders(
            os.path.join('data', 'datasets', 'ustc-tfc2016-pytorch'),
            batch_size=int(args.batch_size),
            n_worker=0,
            train_ratio=train_ratio,
            val_ratio=val_ratio
        )
    else:
        raise NotImplementedError(f"Did not implement for this dataset: {args.dataset}")
    
    
    if 'true' in args.NiN_model:
        print("Training the NiN model")
        model = NiN_CNN1D_TrafficClassification_with_auxiliary(input_ch=1, num_classes=classes).to(device)
    else:
        model = CNN1D_TrafficClassification_with_auxiliary(input_ch=1, num_classes=classes).to(device)
        

    # load best model if it exist / checkpoint
    model_name = model.__class__.__name__
    if args.dataset == "iscx2016vpn":
        inf_model_name = model_name.replace('_with_auxiliary','')
        prev_path = Path(os.path.join('networks', 'pretrained_models', 'iscx2016vpn', (inf_model_name+'_best_model_without_aux.pth')))

    elif args.dataset == "ustctfc2016":
        inf_model_name = model_name.replace('_with_auxiliary','')
        prev_path = Path(os.path.join('networks', 'pretrained_models', 'ustc-tfc2016', (inf_model_name+'_best_model_without_aux.pth')))

    if (prev_path.is_file() == True):
        print(f"Previous trained model exist at {prev_path}\n")
        print("Loading the model...")
        model.load_state_dict(torch.load(prev_path), strict=False)


    # hyperparameter for training
    optimizer = optim.Adam(model.parameters(), lr=float(args.lr))
    #optimizer = optim.Adam(model.parameters(), lr=float(args.lr), betas=(0.95, 0.999))
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

        save_best_model(valid_epoch_loss, epoch, model, optimizer, criterion, args.dataset) # save the best model in this current epoch

    # Save the loss and accuracy plots.
    save_plots(train_acc, valid_acc, train_loss, valid_loss, name=model_name, dataset=args.dataset)
    print(f"TRAINING COMPLETE")
    print(f"Training ratio :{train_ratio}")
    print(f"Val ratio      :{val_ratio}")
    print(f"Test ratio     :{test_ratio}")
    
    # Test accuracy (of unseen test_data) on the best model, using the inference model (without auxiliary)
    if 'true' in args.NiN_model:
        inf_model = NiN_CNN1D_TrafficClassification(input_ch=1, num_classes=classes).to(device)
        with_aux_path = os.path.join('networks', 'pretrained_models', str(args.dataset), (model_name+'_best_model.pth'))
    else:
        inf_model = CNN1D_TrafficClassification(input_ch=1, num_classes=classes).to(device)
        with_aux_path = os.path.join('networks', 'pretrained_models', str(args.dataset), (model_name+'_best_model.pth'))

    model_name = inf_model.__class__.__name__
    checkpoint = torch.load(with_aux_path, map_location=device)
    sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    inf_model.load_state_dict(sd, strict=False)
    test_loss, test_acc = validate(inf_model, test_loader, criterion, device)

    if args.dataset == "iscx2016vpn":
        torch.save(inf_model.state_dict(), os.path.join('networks', 'pretrained_models', 'iscx2016vpn', (model_name+'_best_model_without_aux.pth')))
    
    elif args.dataset == "ustctfc2016":
        torch.save(inf_model.state_dict(), os.path.join('networks', 'pretrained_models', 'ustc-tfc2016', (model_name+'_best_model_without_aux.pth')))
    
    print(f"Acc on unseen test data (not validation dataset): {test_acc}")
