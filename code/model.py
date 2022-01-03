import torch.nn as nn
import torch.nn.functional as F
from gru import GRUCell

nlayers = 2
embedding_size = 512
hidden_size = 512

class RNN(nn.Module):
    def __init__(self, ntoken, dropout):
        super(RNN, self).__init__()

        self.ntoken = ntoken
        self.nlayers = nlayers
        self.input_size = embedding_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(dropout)

        self.encoder = nn.Embedding(ntoken, self.input_size, padding_idx=0)

        self.rnn = nn.ModuleList()
        # N GRU layers
        for i in range(nlayers):
            self.rnn.append(GRUCell(self.input_size, hidden_size))

        self.fc = nn.Linear(hidden_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, ntoken)
        self.init_weights()

    def init_weights(self):
        initrange = 0.05
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        output = self.drop(self.encoder(input))

        for i in range(len(self.rnn)):
            output, hidden[i] = self.rnn[i](output, hidden[i])
            output = self.drop(output)

        output = self.fc(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1,self.ntoken)

        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, batch_size):

        hidden = []
        weight = next(self.parameters())
        for i in range(nlayers):
            hidden.append(weight.new_zeros(batch_size, self.hidden_size))
        return hidden