import torch
import torch.nn as nn
from torchinfo import summary
from models.tinyHAR import TinyHAR_Model


class Transformer(nn.Module):
    def __init__(self, in_size, out_size, win_size, **kwargs):
        super(Transformer, self).__init__()
        # B F L C
        self.model = TinyHAR_Model(
            input_shape = (64, 1, win_size, in_size),
            number_class = out_size,
            filter_num = 20,
            cross_channel_interaction_type = "attn",
            cross_channel_aggregation_type = "FC",
            temporal_info_interaction_type = "lstm",
            temporal_info_aggregation_type = "tnaive"
        )

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        # B, 1, W, C -> B, n_classes
        x = self.model(x)

        # BCE loss
        if len(x.shape) == 1:
            return x.squeeze()
        return x

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        r = self.forward(x)
        return r.argmax(dim=1, keepdim=False)


# Sanity Check
if __name__ == '__main__':
    c, classes, b, w = 6, 11, 128, 100
    m = Transformer(in_size=c, out_size=classes, win_size=w)
    x = torch.rand(b, w, c)

    y = m(x)
    print(f"f's y:{y.shape}")
    print(f"predict:{m.predict(x).shape}")
    summary(m)
