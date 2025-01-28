# from transformers.models.bert.configuration_bert import BertConfig
from transformers import AutoModel,BertConfig,BertModel
from .MCBAM import *
from .MLKA import *

# Build CNN module and CBAM
class CNNNET(nn.Module):
    def __init__(self, input_channel):
        super(CNNNET, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_channel, out_channels=60, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(60),
            nn.GELU())
        self.conv2_1 = nn.Sequential(
            nn.Conv1d(in_channels=60, out_channels=30, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(30),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=30, out_channels=60, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(60),
            nn.GELU())
        self.conv2_2 = nn.Sequential(
            nn.Conv1d(in_channels=60, out_channels=30, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(30),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=30, out_channels=60, kernel_size=3, stride=1, dilation=2, padding=2, bias=False),
            nn.BatchNorm1d(60),
            nn.GELU())
        self.conv2_3 = nn.Sequential(
            nn.Conv1d(in_channels=60, out_channels=30, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(30),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=30, out_channels=60, kernel_size=3, stride=1, dilation=4, padding=4, bias=False),
            nn.BatchNorm1d(60),
            nn.GELU())
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=180, out_channels=180, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),
            nn.BatchNorm1d(180))
        self.ChannelGate = ChannelGate(gate_channels=180, reduction_ratio=12, pool_types=['avg', 'max'])
        self.SpatialGate = SpatialGate()
        self.residual_BN = nn.Sequential(
            nn.Conv1d(180, 180, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),
            nn.BatchNorm1d(180))
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=180, out_channels=180, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(180),
            nn.GELU(),
            nn.Dropout(0.4))
        self.linear1 = nn.Linear(180 * 768, 180)
        self.drop = nn.Dropout(0.5)
        self.linear2 = nn.Linear(180, 2)

    def forward(self, x):
        x = self.conv1(x)
        text1 = self.conv2_1(x)
        text2 = self.conv2_2(x)
        text3 = self.conv2_3(x)
        x = torch.cat([text1, text2, text3], dim=1)
        x = self.conv3(x)
        residual = x
        x = self.ChannelGate(x)
        x = self.SpatialGate(x)
        x = x + self.residual_BN(residual)
        x = self.conv4(x)
        x = x.view(x.shape[0], -1)
        x = self.linear1(x)
        x = self.drop(x)
        x = self.linear2(x)
        return F.softmax(x, dim=1)

# Build BERT-TFBS model
class Bert_Blend_CNN(nn.Module):
    def __init__(self, input_channel,path_preModel:str):
        super(Bert_Blend_CNN, self).__init__()
        config = BertConfig.from_pretrained(path_preModel)
        self.bert = BertModel.from_pretrained(path_preModel, config=config)
        
        # config = BertConfig.from_pretrained("/home/trap/pre-models/DNABERT-2/")
        # self.bert = AutoModel.from_pretrained("/home/trap/pre-models/DNABERT-2/", trust_remote_code=True,config=config)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.model = BERTNew(input_channel)

    def forward(self, X):
        outputs = self.bert(X)
        cls_embeddings = outputs[0]
        cls_embeddings = cls_embeddings[:, 1:-1, :]
        logits = self.model(cls_embeddings)
        return logits


class BERTNewLinear(nn.Module):
    def __init__(self,in_channel):
        super(BERTNewLinear, self).__init__()
        # print("Using BERTNew Model")
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=30, kernel_size=3, stride=1,padding=1,bias=False),
            nn.BatchNorm1d(30),
            nn.GELU(),
            nn.Dropout(p=0.2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=30, out_channels=60, kernel_size=3, stride=1,padding=1,bias=False),
            nn.BatchNorm1d(60),
            nn.GELU(),
            nn.Dropout(p=0.2)
            
        )
        # self.max_pooling_1 = nn.MaxPool1d(kernel_size=4, stride=2)
        # self.lstm = nn.LSTM(input_size=64, hidden_size=21, num_layers=6, bidirectional=True, batch_first=True, dropout=0.2)
        
        self.SpatialGate = SpatialGate()
        self.ChannelGate = ChannelGate(gate_channels=60, reduction_ratio=12, pool_types=['avg', 'max'])
        # self.ChannelGate = ChannelGate(gate_channels=25, reduction_ratio=12, pool_types=['avg', 'max'])

        # self.MLKA = MLKA(in_channels=25, kernel_sizes=[3, 5, 7], dilation=2)
        self.MLKA = MLKA(in_channels=60, kernel_sizes=[3, 5, 7], dilation=2)
        self.conv3 = nn.Conv1d(in_channels=60, out_channels=90, kernel_size=3, stride=1, padding=1, bias=False)

        # self.conv3 = nn.Conv1d(in_channels=60, out_channels=180, kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2)
        self.avgpool = nn.AvgPool1d(kernel_size=3, stride=2)
        # self.combined_pool = CombinedPooling()
        
        self.channel_proj = nn.Conv1d(in_channels=4, out_channels=90, kernel_size=3, stride=1, bias=False)

        self.conv4 = nn.Conv1d(in_channels=90, out_channels=180, kernel_size=3, stride=1, padding=1, bias=False)

        # self.conv4 = nn.Conv1d(in_channels=90, out_channels=180, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=180, out_channels=180, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(180),
            nn.GELU(),
            nn.Dropout(p=0.3)
        )
        self.linear1 = nn.Linear(99 * 768, 99)
        #self.linear1 = nn.Linear(180 * 383, 180)
        self.drop = nn.Dropout(0.3)
        self.linear2 = nn.Linear(99, 2)
        # self.linear2 = nn.Linear(180, 2)
        
        
    
    def forward(self, x):
        # x = x.permute(0, 2, 1)

        #x = self.conv1(x)
        #x = self.conv2(x)
        residual = x
        
        # x = self.SpatialGate(x)
        # x = self.ChannelGate(x)
        
        

        #x1 = self.MLKA(residual)
        #x = self.conv3(x)+self.conv3(x1)
        #avg = x

        #x = self.maxpool(x) 
        #x2 = self.avgpool(avg)
        
        # x_combined = self.combined_pool(x)
        # x_projected = self.channel_proj(x_combined)
        # x = self.conv4(x_projected) 
        #x = self.conv4(x)+self.conv4(x2)
        #x = self.conv5(x)
        x = torch.flatten(x, start_dim=1)  # Flattens from [batch_size, channels, height, width] to [batch_size, channels * height * width]
        x = self.linear1(x)
        x = self.drop(x)
        x = self.linear2(x)
        # x = self.sigmoid(x)
        # return F.sigmoid(x)
        return F.softmax(x,dim=1)
    
    
    
class BERTNew(nn.Module):
    def __init__(self,in_channel):
        super(BERTNew, self).__init__()
        # print("Using BERTNew Model")
        # self.conv1 = nn.Sequential(
        #     nn.Conv1d(in_channels=in_channel, out_channels=30, kernel_size=3, stride=1,padding=1,bias=False),
        #     nn.BatchNorm1d(30),
        #     nn.GELU(),
        #     nn.Dropout(p=0.2)
        # )
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=60, kernel_size=3, stride=1,padding=1,bias=False),
            nn.BatchNorm1d(60),
            nn.GELU(),
            nn.Dropout(p=0.2)
        )
        # self.conv2 = nn.Sequential(
        #     nn.Conv1d(in_channels=30, out_channels=60, kernel_size=3, stride=1,padding=1,bias=False),
        #     nn.BatchNorm1d(60),
        #     nn.GELU(),
        #     nn.Dropout(p=0.2)
            
        # )
        # self.max_pooling_1 = nn.MaxPool1d(kernel_size=4, stride=2)
        # self.lstm = nn.LSTM(input_size=64, hidden_size=21, num_layers=6, bidirectional=True, batch_first=True, dropout=0.2)
        
        self.SpatialGate = SpatialGate()
        self.ChannelGate = ChannelGate(gate_channels=60, reduction_ratio=12, pool_types=['avg', 'max'])
        # self.ChannelGate = ChannelGate(gate_channels=25, reduction_ratio=12, pool_types=['avg', 'max'])

        # self.MLKA = MLKA(in_channels=25, kernel_sizes=[3, 5, 7], dilation=2)
        self.MLKA = MLKA(in_channels=60, kernel_sizes=[3, 5, 7], dilation=2)
        self.conv3 = nn.Conv1d(in_channels=60, out_channels=90, kernel_size=3, stride=1, padding=1, bias=False)

        # self.conv3 = nn.Conv1d(in_channels=60, out_channels=180, kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2)
        self.avgpool = nn.AvgPool1d(kernel_size=3, stride=2)
        # self.combined_pool = CombinedPooling()
        
        self.channel_proj = nn.Conv1d(in_channels=4, out_channels=90, kernel_size=3, stride=1, bias=False)

        self.conv4 = nn.Conv1d(in_channels=90, out_channels=180, kernel_size=3, stride=1, padding=1, bias=False)

        # self.conv4 = nn.Conv1d(in_channels=90, out_channels=180, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=180, out_channels=180, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(180),
            nn.GELU(),
            nn.Dropout(p=0.3)
        )
        # self.linear1 = nn.Linear(180 * 768, 180)
        self.linear1 = nn.Linear(180 * 383, 180)
        self.drop = nn.Dropout(0.3)
        self.linear2 = nn.Linear(180, 2)
        
    
    def forward(self, x):
        # x = x.permute(0, 2, 1)
       
        x = self.conv1(x)
        # x = self.conv2(x)
        residual = x
        
        x = self.SpatialGate(x)
        x = self.ChannelGate(x)
       
        x1 = self.MLKA(residual)
        x = self.conv3(x)+self.conv3(x1)
        avg = x
       
        x = self.maxpool(x) 
        x2 = self.avgpool(avg)
        
        # x_combined = self.combined_pool(x)
        # x_projected = self.channel_proj(x_combined)
        # x = self.conv4(x_projected) 
        x = self.conv4(x)+self.conv4(x2)
        x = self.conv5(x)
        x = torch.flatten(x, start_dim=1)  # Flattens from [batch_size, channels, height, width] to [batch_size, channels * height * width]
        x = self.linear1(x)
        x = self.drop(x)
        x = self.linear2(x)
        # x = self.sigmoid(x)
        # return F.sigmoid(x)
        return F.softmax(x,dim=1)