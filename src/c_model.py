import torch.nn as nn
import torch
import torch.nn.functional as F

class RNN(nn.Module):

    def __init__(
        self,
        embed_dim=200,
        hidden_size=100,
        n_layers=3,
        dropout=0.2,
    ):

        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        super().__init__()

        self.rnn = nn.LSTM(
            input_size=self.embed_dim,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            dropout=self.dropout,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):

        # |x| = (batch_size, length, embed_dim)
        x, _ = self.rnn(x)

        y = x[:, -1]

        return y

class C_rnn(nn.Module):

    def __init__(
        self,
        input_size=None,
        embed_dim=200,
        hidden_size=100,
        n_classes=2,
        n_layers=3,
        dropout=0.2,
        pretrained_embedding=None,
        freeze_embedding=False,):

        super(C_rnn, self).__init__()

        self.input_size = input_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.dropout = dropout
        self.pretrained_embedding = pretrained_embedding
        self.freeze_embedding = freeze_embedding

        if pretrained_embedding is not None:
            print("doing with pretrained model!!!")
            self.input_size, self.embed_dim = pretrained_embedding.shape
            self.emb = nn.Embedding.from_pretrained(pretrained_embedding,
                                                    freeze=freeze_embedding)
        else:
            print("doing without pretrained model!!!")
            self.embed_dim = embed_dim
            self.emb = nn.Embedding(input_size, embed_dim)
        
        self.lstm1 = RNN(
            embed_dim = self.embed_dim,
            hidden_size = self.hidden_size,
            n_layers = self.n_layers,
            dropout = self.dropout
        )

        self.lstm2 = RNN(
            embed_dim = self.embed_dim,
            hidden_size = self.hidden_size,
            n_layers = self.n_layers,
            dropout = self.dropout
        )

        self.fc1 = nn.Linear(
            self.hidden_size*2*2,
            # self.n_classes
            300
        )

        self.fc2 = nn.Linear(
            300, 
            self.n_classes
        )

        self.dp1 = nn.Dropout(self.dropout)
        self.dp2 = nn.Dropout(self.dropout)
    
    def forward(self, x1, x2):

        a = self.emb(x1).float()
        b = self.emb(x2).float()

        a = self.lstm1(a)
        b = self.lstm2(b)

        y = torch.cat((a, b), 1)

        y = self.fc1(self.dp1(F.relu(y)))
        y = self.fc2(self.dp2(F.relu(y)))

        return y