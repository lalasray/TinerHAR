import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, repr_size, n_classes, drp_rate=0.2):
        super(MLPClassifier, self).__init__()
        self.mlp = nn.Sequential(
            self.block(repr_size, 256, drp_rate),
            self.block(256, 128, drp_rate),
            self.block(128, n_classes, 0.0)
        )

    def block(self, cin, cout, drp):
        blk = nn.Sequential(
            nn.Linear(cin, cout),
            nn.BatchNorm1d(cout),
            nn.ReLU(inplace=True),
            nn.Dropout(drp)
        )
        return blk

    def forward(self, x):
        return self.mlp(x)


class ClassicEncoderCPC(nn.Module):
    def __init__(self, in_size, k_size=5):
        super(ClassicEncoderCPC, self).__init__()
        lst_filters = [32, 64, 128]
        dropout = 0.2
        self.convs = nn.Sequential(
            nn.Conv1d(in_size, lst_filters[0], kernel_size=k_size, padding='same'),
            nn.Dropout(dropout),
            nn.ReLU(True),

            nn.Conv1d(lst_filters[0], lst_filters[1], kernel_size=k_size, padding='same'),
            nn.Dropout(dropout),
            nn.ReLU(True),

            nn.Conv1d(lst_filters[1], lst_filters[2], kernel_size=k_size, padding='same'),
            nn.Dropout(dropout),
            nn.ReLU(True)
        )
        self.rnn = nn.GRU(input_size=lst_filters[-1], num_layers=2, hidden_size=256, batch_first=True, dropout=dropout)

    def forward(self, x, pretext=False):
        z = self.convs(x).permute(0, 2, 1)
        outputs, h_n_last = self.rnn(z)
        if pretext:
            return z, outputs, h_n_last
        # D*layers, N, C
        return h_n_last[-1, :]


class ClassificationWithEncoderCPC(nn.Module):
    def __init__(self, in_size, out_size, encoder_weights_path, repr_size=256):
        super(ClassificationWithEncoderCPC, self).__init__()
        encoder = ClassicEncoderCPC(in_size=in_size)
        encoder.load_state_dict(torch.load(encoder_weights_path,map_location=torch.device('cpu')))
        self.encoder = encoder
        self.classifier = MLPClassifier(repr_size=repr_size, n_classes=out_size)

    def forward(self, x):
        x_feats = self.encoder(x.permute(0,2,1))
        logits = self.classifier(x_feats)
        return logits


class ContrastivePredictiveCoding(nn.Module):
    def __init__(self, in_size, num_steps_prediction=28, repr_size=256, **kwargs):
        super(ContrastivePredictiveCoding, self).__init__()

        # 1D conv Encoder to get the outputs at each timestep
        self.encoder = ClassicEncoderCPC(in_size=in_size)

        # RNN to obtain context vector
        #self.rnn = nn.GRU(input_size=repr_size, hidden_size=256, num_layers=2, bidirectional=False, batch_first=True, dropout=0.2)

        # Projections for k steps
        conv_size = 128
        self.Wk = nn.ModuleList([nn.Linear(repr_size, conv_size) for i in range(num_steps_prediction)])

        # Softmaxes for the loss computation
        self.softmax = nn.Softmax(dim=1)
        self.lsoftmax = nn.LogSoftmax(dim=1)

        # Other details
        self.num_steps_prediction = num_steps_prediction

    def forward(self, inputs):
        inputs = inputs.permute(0,2,1)
        # Passing through the encoder. Input: BxCxT and output is: Bx128XT
        # z = self.encoder(inputs, pretext=True)
        if type(self.encoder) == ClassicEncoderCPC:
            z = self.encoder.convs(inputs).permute(0,2,1)

            # Random timestep to start the future prediction from.
            # If the window is 50 timesteps and k=12, we pick a number from 0-37
            start = torch.randint(int(inputs.shape[2] - self.num_steps_prediction), size=(1,)).long()

            # Need to pick the encoded data only until the starting timestep
            rnn_input = z[:, :start + 1, :]

            # Passing through the RNN
            r_out, (_) = self.encoder.rnn(rnn_input, None)

        return z, r_out, start

    def compute_cpc_loss(self, z, c, t):
        batch_size = z.shape[0]

        # The context vector is the last timestep from the RNN
        c_t = c[:, t, :].squeeze(1)

        # infer z_{t+k} for each step in the future: c_t*Wk, where 1 <= k <= timestep
        pred = torch.stack([self.Wk[k](c_t) for k in range(self.num_steps_prediction)])

        # pick the target z values k timestep number of samples after t
        z_samples = z[:, t + 1: t + 1 + self.num_steps_prediction, :].permute(1, 0, 2)

        nce = 0


        # Looping over the number of timesteps chosen
        for k in range(self.num_steps_prediction):
            # calculate the log density ratio: log(f_k) = z_{t+k}^T * W_k * c_t
            log_density_ratio = torch.mm(z_samples[k], pred[k].transpose(0, 1))

            # calculate NCE loss
            nce = nce + torch.sum(torch.diag(self.lsoftmax(log_density_ratio)))

        # average over timestep and batch
        nce = nce / (-1.0 * batch_size * self.num_steps_prediction)
        return nce

    def predict_features(self, inputs):
        z = self.encoder(inputs)

        # Passing through the RNN
        r_out, _ = self.encoder.rnn(z, None)

        return r_out

    def get_loss(self, inputs):
        z, r_out, start = self.forward(inputs)
        nce = self.compute_cpc_loss(z, r_out, start)

        return nce

    def forward_classification(self, x):
        x_encoded = self.encoder(x)
        return self.classifier(x_encoded)