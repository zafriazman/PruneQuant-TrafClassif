import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

# Written according to the paper 
#   TWO-STAGE DISTILLATION-AWARE COMPRESSED MODELS FOR TRAFFIC CLASSIFICATION


class NiN_CNN1D_TrafficClassification_with_auxiliary(nn.Module):
    """
    1-D CNN for the iscx2016vpn dataset
    input_feature_length is the number of features from the input (preprocessed iscx2016vpn dataset has 1500x1 normalized byte as the features)
    """
    def __init__(self, input_ch, num_classes):
        super().__init__()
        
        # Block 1
        self.conv1 = nn.Conv1d(in_channels=input_ch, out_channels=16, kernel_size=2, stride=2, padding=0)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm1d(16)
        self.submod1 = nn.Conv1d(in_channels=16, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        
        # Block 2
        self.conv4 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2, stride=2, padding=0)
        self.bn3 = nn.BatchNorm1d(32)
        self.conv5 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv6 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.bn4 = nn.BatchNorm1d(32)
        self.submod2 = nn.Conv1d(in_channels=32, out_channels=num_classes, kernel_size=1, stride=1, padding=0)

        # Block 3
        self.conv7 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2, stride=2, padding=0)
        self.bn5 = nn.BatchNorm1d(64)
        self.conv8 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv9 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn6 = nn.BatchNorm1d(64)
        self.submod3 = nn.Conv1d(in_channels=64, out_channels=num_classes, kernel_size=1, stride=1, padding=0)

        self.conv10 = nn.Conv1d(in_channels=64, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        
        # Generic layer
        self.relu = nn.ReLU()

    
    def forward(self, x):
        auxiliary_outputs = []

        # Block 1
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.conv2(x))
        x = self.relu(self.bn2(self.conv3(x)))
        if self.training:
            auxiliary = self.submod1(x)
            auxiliary_outputs.append(self.process_auxiliary(auxiliary))

        # Block 2
        x = self.relu(self.bn3(self.conv4(x)))
        x = self.relu(self.conv5(x))
        x = self.relu(self.bn4(self.conv6(x)))
        if self.training:
            auxiliary = self.submod2(x)
            auxiliary_outputs.append(self.process_auxiliary(auxiliary))

        # Block 3
        x = self.relu(self.bn5(self.conv7(x)))
        x = self.relu(self.conv8(x))
        x = self.relu(self.bn6(self.conv9(x)))
        if self.training:
            auxiliary = self.submod3(x)
            auxiliary_outputs.append(self.process_auxiliary(auxiliary))
        
        x = self.conv10(x)
        x = F.avg_pool1d(x, x.size(2))  # Apply Global Average Pooling (GAP) to prepare for classification
        x = x.reshape(x.shape[0], -1)   # Flatten

        if self.training:
            return x, auxiliary_outputs
        return x

    def process_auxiliary(self, x):
        x = F.avg_pool1d(x, x.size(2))
        x = x.reshape(x.shape[0], -1)  # Flatten
        return x

class NiN_CNN1D_TrafficClassification(nn.Module):
    """
    Same as "NiN_CNN1D_TrafficClassification_with_auxiliary" class,
    but without the auxiliary learner (submod)
    """
    def __init__(self, input_ch, num_classes):
        super().__init__()
        
        # Block 1
        self.conv1 = nn.Conv1d(in_channels=input_ch, out_channels=16, kernel_size=2, stride=2, padding=0)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm1d(16)
        
        # Block 2
        self.conv4 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2, stride=2, padding=0)
        self.bn3 = nn.BatchNorm1d(32)
        self.conv5 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv6 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.bn4 = nn.BatchNorm1d(32)

        # Block 3
        self.conv7 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2, stride=2, padding=0)
        self.bn5 = nn.BatchNorm1d(64)
        self.conv8 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv9 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn6 = nn.BatchNorm1d(64)

        self.conv10 = nn.Conv1d(in_channels=64, out_channels=num_classes, kernel_size=1, stride=1, padding=0)

        self.relu = nn.ReLU()

    
    def forward(self, x):

        # Block 1
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.conv2(x))
        x = self.relu(self.bn2(self.conv3(x)))

        # Block 2
        x = self.relu(self.bn3(self.conv4(x)))
        x = self.relu(self.conv5(x))
        x = self.relu(self.bn4(self.conv6(x)))

        # Block 3
        x = self.relu(self.bn5(self.conv7(x)))
        x = self.relu(self.conv8(x))
        x = self.relu(self.bn6(self.conv9(x)))
        
        x = self.conv10(x)
        x = F.avg_pool1d(x, x.size(2))
        x = x.reshape(x.shape[0], -1)

        return x
    


class NiN_CNN1D_TrafficClassification_Prune30Percent(nn.Module):
    """
    Same as "NiN_CNN1D_TrafficClassification" class (without auxiliary learning),
    But modified to have 30% less parameter (50% FLOPs reduction according to NiN paper).
    This is to emulate the speed if it has been pruned (The accuracy is not discussed using this model).
    """
    def __init__(self, input_ch, num_classes):
        super().__init__()
        
        # Block 1
        self.conv1 = nn.Conv1d(in_channels=input_ch, out_channels=11, kernel_size=2, stride=2, padding=0)
        self.bn1 = nn.BatchNorm1d(11)
        self.conv2 = nn.Conv1d(in_channels=11, out_channels=11, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv1d(in_channels=11, out_channels=11, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm1d(11)
        
        # Block 2
        self.conv4 = nn.Conv1d(in_channels=11, out_channels=22, kernel_size=2, stride=2, padding=0)
        self.bn3 = nn.BatchNorm1d(22)
        self.conv5 = nn.Conv1d(in_channels=22, out_channels=22, kernel_size=1, stride=1, padding=0)
        self.conv6 = nn.Conv1d(in_channels=22, out_channels=22, kernel_size=1, stride=1, padding=0)
        self.bn4 = nn.BatchNorm1d(22)

        # Block 3
        self.conv7 = nn.Conv1d(in_channels=22, out_channels=44, kernel_size=2, stride=2, padding=0)
        self.bn5 = nn.BatchNorm1d(44)
        self.conv8 = nn.Conv1d(in_channels=44, out_channels=44, kernel_size=1, stride=1, padding=0)
        self.conv9 = nn.Conv1d(in_channels=44, out_channels=44, kernel_size=1, stride=1, padding=0)
        self.bn6 = nn.BatchNorm1d(44)

        self.conv10 = nn.Conv1d(in_channels=44, out_channels=num_classes, kernel_size=1, stride=1, padding=0)

        self.relu = nn.ReLU()

    
    def forward(self, x):

        # Block 1
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.conv2(x))
        x = self.relu(self.bn2(self.conv3(x)))

        # Block 2
        x = self.relu(self.bn3(self.conv4(x)))
        x = self.relu(self.conv5(x))
        x = self.relu(self.bn4(self.conv6(x)))

        # Block 3
        x = self.relu(self.bn5(self.conv7(x)))
        x = self.relu(self.conv8(x))
        x = self.relu(self.bn6(self.conv9(x)))
        
        x = self.conv10(x)
        x = F.avg_pool1d(x, x.size(2))
        x = x.reshape(x.shape[0], -1)

        return x



# Get the model summary when given 1, 1500 input
def get_model_summary(model):
    """
    get model parameters, keras style
    """
    summary(model, input_size=(1,1500))



if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = NiN_CNN1D_TrafficClassification_with_auxiliary(input_ch=1, num_classes=16).to(device)
    inf_model = NiN_CNN1D_TrafficClassification(input_ch=1, num_classes=16).to(device)

    input_x = torch.rand([1024, 1, 1500]).to(device)   # input to a conv1d model is [batch_size, channel, length]
    main_output, auxiliary_outputs = model(input_x)
    output = inf_model(input_x)

    print(f"Final output shape of the model: {output.shape}")

    print("Model summary for training model")
    get_model_summary(model)
    
    print("\n\nModel summary for inferencing model")
    get_model_summary(inf_model)