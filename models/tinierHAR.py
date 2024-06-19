import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
import time
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func


'''
class SelfAttention_interaction(nn.Module):
    def __init__(self, sensor_channel, n_channels):
        super(SelfAttention_interaction, self).__init__()

        self.query = nn.Linear(n_channels, n_channels, bias=False)
        self.key = nn.Linear(n_channels, n_channels, bias=False)
        self.value = nn.Linear(n_channels, n_channels, bias=False)
        self.gamma = nn.Parameter(torch.tensor([0.], dtype=torch.float32))
        
    def forward(self, x):
        # Ensure input is in float16 and move to GPU
        x = x.to(torch.float16).cuda()
        
        # Ensure weights are in float16 and move to GPU
        self.query.weight.data = self.query.weight.data.to(torch.float16).cuda()
        self.key.weight.data = self.key.weight.data.to(torch.float16).cuda()
        self.value.weight.data = self.value.weight.data.to(torch.float16).cuda()
        
        # Project input to Q, K, V
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Reshape to (batch, seqlen, num_heads, head_dim)
        batch_size, seqlen, feature_dim = x.shape
        num_heads = 1  # Since we are not using multi-head attention here
        head_dim = feature_dim // num_heads
        
        q = q.view(batch_size, seqlen, num_heads, head_dim)
        k = k.view(batch_size, seqlen, num_heads, head_dim)
        v = v.view(batch_size, seqlen, num_heads, head_dim)
        
        # Apply FlashAttention
        qkv = torch.stack((q, k, v), dim=2)  # Shape: (batch_size, seqlen, 3, num_heads, head_dim)
        out = flash_attn_qkvpacked_func(qkv, dropout_p=0.0, causal=False)
        
        # Reshape the output back to (batch, seqlen, feature_dim)
        out = out.view(batch_size, seqlen, feature_dim)
        
        # Convert gamma to float16 and move to GPU if needed
        gamma = self.gamma.to(torch.float16).cuda()
        
        # Apply gamma parameter
        out = gamma * out + x
        
        # Convert the output back to float32 if needed and move to CPU
        out = out.to(torch.float32).cpu()
        
        return out

'''
class SelfAttention_interaction(nn.Module):
    """

    """

    def __init__(self, sensor_channel, n_channels):
        super(SelfAttention_interaction, self).__init__()

        self.query = nn.Linear(n_channels, n_channels, bias=False)
        self.key = nn.Linear(n_channels, n_channels, bias=False)
        self.value = nn.Linear(n_channels, n_channels, bias=False)
        self.gamma = nn.Parameter(torch.tensor([0.]))

        # self.fc1            = nn.Linear(n_channels, n_channels, bias=False)
        # self.fc_activation = nn.ReLU()
        # self.fc2            = nn.Linear(n_channels, n_channels, bias=False)
        # self.beta         = nn.Parameter(torch.tensor([0.]))

    def forward(self, x):
        # 输入尺寸是 batch  sensor_channel feature_dim
        # print(x.shape)

        f, g, h = self.query(x), self.key(x), self.value(x)

        beta = F.softmax(torch.bmm(f, g.permute(0, 2, 1).contiguous()), dim=1)

        o = self.gamma * torch.bmm(h.permute(0, 2, 1).contiguous(), beta) + x.permute(0, 2, 1).contiguous()
        o = o.permute(0, 2, 1).contiguous()

        # o = self.beta  * self.fc2(self.fc_activation(self.fc1(o)))  +  o
        # 输出是 batch  sensor_channel feature_dim 1
        return o


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=16, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer_interaction(nn.Module):
    def __init__(self, sensor_channel, dim, depth=1, heads=4, dim_head=16, mlp_dim=16, dropout=0.):
        super(Transformer_interaction, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return x
    

# Define the parameters
sensor_channel = 10  # Example sensor_channel dimension
n_channels = 64      # Example n_channels dimension

# Instantiate the model
model = SelfAttention_interaction(sensor_channel, n_channels).cuda()


# Measure total parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

# Create a dummy input tensor with the shape (batch, sensor_channel, feature_dim)
batch_size = 8        # Example batch size
feature_dim = 64      # Example feature dimension
dummy_input = torch.randn(batch_size, sensor_channel, feature_dim).cuda()


# Measure the time taken for a forward pass
start_time = time.time()
output = model(dummy_input)
end_time = time.time()

# Print the time taken
print(f"Time taken for forward pass: {end_time - start_time} seconds")

# Print the shapes
print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}")

# Manually calculate FLOPs
def calculate_flops(model, input_tensor):
    flops = 0

    # Convert input_tensor to float16 and move to GPU
    input_tensor = input_tensor.to(torch.float16).cuda()

    # Linear layer FLOPs: 2 * input_features * output_features
    flops += 2 * input_tensor.size(1) * model.query.out_features  # query
    flops += 2 * input_tensor.size(1) * model.key.out_features    # key
    flops += 2 * input_tensor.size(1) * model.value.out_features  # value

    # Matrix multiplication FLOPs: 2 * M * N * K
    f = model.query(input_tensor)
    g = model.key(input_tensor)
    h = model.value(input_tensor)

    batch_size, seq_length, n_channels = f.size()
    flops += 2 * batch_size * seq_length * n_channels * seq_length  # torch.bmm(f, g.permute(0, 2, 1))
    flops += 2 * batch_size * n_channels * seq_length * seq_length  # torch.bmm(h.permute(0, 2, 1), beta)

    # Softmax FLOPs: 2 * input_size
    flops += 2 * batch_size * seq_length * seq_length

    # Element-wise multiplication and addition FLOPs
    flops += batch_size * seq_length * n_channels  # gamma * torch.bmm(...)
    flops += batch_size * seq_length * n_channels  # + x.permute(...)

    return flops


total_flops = calculate_flops(model, dummy_input)
print(f"Total FLOPs: {total_flops}")