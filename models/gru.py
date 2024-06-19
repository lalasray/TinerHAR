import torch
from torch import nn
from torchinfo import summary


class GRU(torch.nn.Module):

    def __init__(self, input_dim, output_dim, window_size, hidden_dim=128, layer_num=2):
        super(GRU, self).__init__()
        self.gru1 = nn.GRU(input_dim, hidden_dim, layer_num, batch_first=True)
        self.gru2 = nn.GRU(hidden_dim, hidden_dim, layer_num, batch_first=True)
        self.dense = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)
        self.bn = nn.BatchNorm1d(window_size)

    def forward(self, inputs):
        x = self.bn(inputs)
        x, hf = self.gru1(x)
        x, hf = self.gru2(x, hf)
        x = self.dropout(x)
        out = self.dense(x[:, -1, :])
        return out

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        logits = self.forward(x)
        return logits.argmax(dim=-1)


# Sanity Check
if __name__ == '__main__':
    c, classes, b, w = 6, 11, 72, 400
    m = GRU(c, classes, w)
    x = torch.rand(b, w, c)

    y = m(x)
    print(f"f's y:{y.shape}")
    print(f"predict:{m.predict(x).shape}")
    summary(m)
