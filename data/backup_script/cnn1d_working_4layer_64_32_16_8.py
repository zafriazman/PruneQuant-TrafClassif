import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary



class CNN1D_TrafficClassification(nn.Module):
    """
    1-D CNN for the iscx2016vpn dataset
    input_feature_length is the number of features from the input (preprocessed iscx2016vpn dataset has 1500x1 normalized byte as the features)
    """
    def __init__(self, input_ch, num_classes):
        super().__init__()
        
        # Layer 1
        # input [batch_size, 1, 1500]  -->  afterconv&BN [batch_size, 64, 1500]   -->  afterpool [batch_size, 64, 750]
        self.conv1 = nn.Conv1d(in_channels=input_ch, out_channels=64, kernel_size=3, stride=1, padding='same')
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Layer 2
        # input [batch_size, 64, 750]  -->  afterconv&BN [batch_size, 32, 750]   -->  afterpool [batch_size, 32, 375]
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding='same')
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Layer 3
        # input [batch_size, 32, 375]  -->  afterconv&BN [batch_size, 16, 375]   -->  afterpool [batch_size, 16, 187]
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding='same')
        self.bn3 = nn.BatchNorm1d(16)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Layer 4
        # input [batch_size, 16, 187]  -->  afterconv&BN [batch_size, 8, 187]   -->  afterpool [batch_size, 8, 93]
        self.conv4 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding='same')
        self.bn4 = nn.BatchNorm1d(8)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        # FC layer.
        # input is expected as a flatten [batch_size, 16, 187]
        self.fc = nn.Linear(8*93, num_classes) # num_classes in iscx2016vpn is 16
        
        # Generic layer
        self.relu = nn.ReLU()

    
    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu(self.bn4(self.conv4(x))))
        x = x.reshape(x.shape[0], -1)   # Flatten all dimension except batch dimension
        x = self.fc(x)
        
        return x



# Get the model summary when given 1, 1500 input
def get_model_summary(model):
    """
    get model parameters, keras style
    """
    summary(model, input_size=(1,1500))



if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = CNN1D_TrafficClassification(input_ch=1, num_classes=16).to(device)

    input_x = torch.rand([1024, 1, 1500]).to(device)   # input to a conv1d model is [batch_size, channel, length]
    output = model(input_x)

    print(f"Final output shape of the model: {output.shape}")

    get_model_summary(model)
    