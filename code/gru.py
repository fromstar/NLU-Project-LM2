import torch.nn as nn
import torch

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.update_gate = nn.Linear(hidden_size * 2, hidden_size)
        self.reset_gate = nn.Linear(hidden_size * 2, hidden_size)
        self.out_gate = nn.Linear(hidden_size, hidden_size)
        self.x = nn.Linear(hidden_size, hidden_size)

    def forward(self, input, prev_state):

        seq_size, _, _ = input.size()
        hidden_seq = []

        for t in range(seq_size):

            x_t = input[t, :, :]
            x_h = torch.cat((x_t, prev_state), dim=1)

            reset = torch.sigmoid(self.reset_gate(x_h))
            update = torch.sigmoid(self.update_gate(x_h))

            n1 = self.out_gate(prev_state) * reset
            n2 = n1 + self.x(x_t)
            out = torch.tanh(n2)

            new_state = (1 - update) * out + update * prev_state
            hidden_seq.append(new_state.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)

        return hidden_seq, new_state