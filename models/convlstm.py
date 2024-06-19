import torch
import torch.nn as nn
from torchinfo import summary


class ConvLSTM(nn.Module):

    def __init__(self, in_size=3, out_size=10, conv_features=64, kernel_size=5, LSTM_units=128, **kwargs):
        super(ConvLSTM, self).__init__()

        self.conv1 = nn.Conv2d(1, conv_features, (kernel_size, 1))
        self.conv2 = nn.Conv2d(conv_features, conv_features, (kernel_size, 1))
        self.conv3 = nn.Conv2d(conv_features, conv_features, (kernel_size, 1))
        self.conv4 = nn.Conv2d(conv_features, conv_features, (kernel_size, 1))

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(in_size * conv_features, LSTM_units, num_layers=2)
        self.classifier = nn.Linear(LSTM_units, out_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))

        x = x.permute(2, 0, 3, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        x = self.dropout(x)

        x, h = self.lstm(x)
        x = x[-1, :, :]

        out = self.classifier(x)

        return out

    @torch.no_grad()
    def predict(self, x):
        logits = self.forward(x)
        return logits.argmax(dim=-1)


if __name__ == '__main__':
    c, classes, b, w = 6, 11, 128, 100
    m = ConvLSTM(c, classes)
    x = torch.rand(b, w, c)

    y = m(x)
    print(f"f's y:{y.shape}  -- x.shape: {x.shape}")
    print(f"predict:{m.predict(x).shape}")
    summary(m)