import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class CNN1D_TrafficClassification_with_auxiliary(nn.Module):
    """
    1-D CNN for the iscx2016vpn dataset
    input_feature_length is the number of features from the input (preprocessed iscx2016vpn dataset has 1500x1 normalized byte as the features)
    """
    def __init__(self, input_ch, num_classes):
        super().__init__()
        
        # Layer 1
        # input [batch_size, 1, 1500]  -->  afterconv&BN [batch_size, 16, 1500]   -->  afterpool [batch_size, 16, 500]
        self.conv1 = nn.Conv1d(in_channels=input_ch, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=3)

        self.submod1 = nn.Conv1d(in_channels=16, out_channels=num_classes, kernel_size=1, stride=1, padding=0)

        # Layer 2
        # input [batch_size, 16, 500]  -->  afterconv&BN [batch_size, 16, 500]   -->  afterpool [batch_size, 16, 166]
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(16)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=3)

        self.submod2 = nn.Conv1d(in_channels=16, out_channels=num_classes, kernel_size=1, stride=1, padding=0)

        # Layer 3
        # input [batch_size, 16, 166]  -->  afterconv&BN [batch_size, 32, 166]   -->  afterpool [batch_size, 32, 83]
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.submod3 = nn.Conv1d(in_channels=32, out_channels=num_classes, kernel_size=1, stride=1, padding=0)

        # Layer 4
        # input [batch_size, 32, 83]  -->  afterconv&BN [batch_size, 32, 83]   -->  afterpool [batch_size, 32, 41]
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(32)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.submod4 = nn.Conv1d(in_channels=32, out_channels=num_classes, kernel_size=1, stride=1, padding=0)

        # Layer 5
        # input [batch_size, 32, 41]  -->  afterconv&BN [batch_size, num_classess(16), 41]
        self.conv5 = nn.Conv1d(in_channels=32, out_channels=num_classes, kernel_size=1, stride=1, padding=0)

        # FC layer.
        # input is expected as a flatten [batch_size, 16, 187]
        #self.fc1 = nn.Linear(32*41, num_classes) # num_classes in iscx2016vpn is 16
        
        # Generic layer
        self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(p=0.01)

    
    def forward(self, x):

        auxiliary_classification = []
        auxiliary_fmap = []

        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        #x = self.dropout(x)
        if self.training:
            auxiliary = self.submod1(x)
            auxiliary_fmap.append(auxiliary)
            auxiliary_classification.append(self.process_auxiliary(auxiliary))

        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        #x = self.dropout(x)
        if self.training:
            auxiliary = self.submod2(x)
            auxiliary_fmap.append(auxiliary)
            auxiliary_classification.append(self.process_auxiliary(auxiliary))

        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        #x = self.dropout(x)
        if self.training:
            auxiliary = self.submod3(x)
            auxiliary_fmap.append(auxiliary)
            auxiliary_classification.append(self.process_auxiliary(auxiliary))

        x = self.pool4(self.relu(self.bn4(self.conv4(x))))
        #x = self.dropout(x)
        if self.training:
            auxiliary = self.submod4(x)
            auxiliary_fmap.append(auxiliary)
            auxiliary_classification.append(self.process_auxiliary(auxiliary))

        x = self.conv5(x)
        auxiliary_fmap.append(x)

        # Apply Global Average Pooling (GAP)
        # Using avg_pool1d with kernel size equal to the length of the feature maps
        #x = F.avg_pool1d(x, x.size(2))
        # x.size(2) is a condition statement. It is not symbolically traceable, thus cannot use as jit compilation (for quant)

        x = torch.mean(x, dim=-1, keepdim=True) # manual avg_pool1d on the last dimension (channel)
        x = x.reshape(x.shape[0], -1)           # Flatten all dimension except batch dimension
        #x = self.fc1(x)                        # No need FC since we are using GAP
        auxiliary_classification.append(x)

        if self.training:
            return auxiliary_classification, auxiliary_fmap
        return x

    def process_auxiliary(self, x):
        #x = F.avg_pool1d(x, x.size(2))
        x = torch.mean(x, dim=-1, keepdim=True)
        x = x.reshape(x.shape[0], -1)  # Flatten
        return x

class CNN1D_TrafficClassification(nn.Module):
    """
    Same as "CNN1D_TrafficClassification_with_auxiliary" class,
    but without the auxiliary learner (submod)
    """
    def __init__(self, input_ch, num_classes):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels=input_ch, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=3)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(16)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=3)

        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(32)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv1d(in_channels=32, out_channels=num_classes, kernel_size=1, stride=1, padding=0)

        self.relu = nn.ReLU()

    
    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu(self.bn4(self.conv4(x))))
        x = self.conv5(x)
        #x = F.avg_pool1d(x, x.size(2))
        x = torch.mean(x, dim=-1, keepdim=True)
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
    model = CNN1D_TrafficClassification_with_auxiliary(input_ch=1, num_classes=16).to(device)
    inf_model = CNN1D_TrafficClassification(input_ch=1, num_classes=16).to(device)

    input_x = torch.rand([1024, 1, 1500]).to(device)   # input to a conv1d model is [batch_size, channel, length]
    main_output, auxiliary_classification = model(input_x)
    output = inf_model(input_x)

    print(f"Final output shape of the model: {output.shape}")

    print("Model summary for training model")
    get_model_summary(model)
    
    print("\n\nModel summary for inferencing model")
    get_model_summary(inf_model)
    