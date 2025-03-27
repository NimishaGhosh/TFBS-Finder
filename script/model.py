# from transformers.models.bert.configuration_bert import BertConfig
from transformers import AutoModel,BertConfig,BertModel
from .MCBAM import *
from .MLKA import *


    
class BERTNew(nn.Module):
    def __init__(self,in_channel):
        super(BERTNew, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=60, kernel_size=3, stride=1,padding=1,bias=False),
            nn.BatchNorm1d(60),
            nn.GELU(),
            nn.Dropout(p=0.2)
        )
        self.SpatialGate = SpatialGate()
        self.ChannelGate = ChannelGate(gate_channels=60, reduction_ratio=12, pool_types=['avg', 'max'])

        self.MLKA = MLKA(in_channels=60, kernel_sizes=[3, 5, 7], dilation=2)
        self.conv3 = nn.Conv1d(in_channels=60, out_channels=90, kernel_size=3, stride=1, padding=1, bias=False)

        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2)
        self.avgpool = nn.AvgPool1d(kernel_size=3, stride=2)
        

        self.conv4 = nn.Conv1d(in_channels=90, out_channels=180, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=180, out_channels=180, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(180),
            nn.GELU(),
            nn.Dropout(p=0.3)
        )
        self.linear1 = nn.Linear(180 * 383, 180)
        self.drop = nn.Dropout(0.3)
        self.linear2 = nn.Linear(180, 2)
        
    
    def forward(self, x):
        # x = x.permute(0, 2, 1)
       
        x = self.conv1(x)
        residual = x
        
        x = self.SpatialGate(x)
        x = self.ChannelGate(x)
       
        x1 = self.MLKA(residual)
        x = self.conv3(x)+self.conv3(x1)
        avg = x
       
        x = self.maxpool(x) 
        x2 = self.avgpool(avg)
        
  
        x = self.conv4(x)+self.conv4(x2)
        x = self.conv5(x)
        x = torch.flatten(x, start_dim=1)  # Flattens from [batch_size, channels, height, width] to [batch_size, channels * height * width]
        x = self.linear1(x)
        x = self.drop(x)
        x = self.linear2(x)

        return F.softmax(x,dim=1)

class Bert_MCBAM_MLKA(nn.Module):
    def __init__(self, input_channel):
        super(Bert_MCBAM_MLKA, self).__init__()
        self.bert = BertModel.from_pretrained("DNABERT5", trust_remote_code=True)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.model = BERTNew(input_channel)

    def forward(self, X):
        

        outputs = self.bert(X)
        cls_embeddings = outputs[0]
        cls_embeddings = cls_embeddings[:, 1:-1, :]
        logits = self.model(cls_embeddings)
        return logits
