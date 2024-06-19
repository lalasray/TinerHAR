import torch
from torch import nn
from torchinfo import summary


class LSTM(nn.Module):
    def __init__(self, input_dim, output_dim, window_size, hidden_dim=128, layer_num=2):
        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, layer_num, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, layer_num, batch_first=True)
        self.dense = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)
        self.bn = nn.BatchNorm1d(window_size)

    def forward(self, inputs):
        x = self.bn(inputs)
        x, hf = self.lstm1(x)
        x, hf = self.lstm2(x, hf)
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
    m = LSTM(c, classes, w)
    x = torch.rand(b, w, c)

    y = m(x)
    print(f"f's y:{y.shape}")
    print(f"predict:{m.predict(x).shape}")
    summary(m)
