import torch
import torch.nn.functional as F

def smooth_data(data, window_size):
    padded_data = torch.cat([data[:window_size//2], data, data[-window_size//2:]])
    smoothed_data = torch.zeros_like(data)
    for i in range(data.shape[0]):
        smoothed_data[i] = padded_data[i:i+window_size].mean()
    return smoothed_data

def generate_data(n_samples, g_x_true=10.0, alpha_x_true=13.0, beta_x_true=0.7, window_size=5):
    t = torch.linspace(0, 100, n_samples).view(-1, 1)
    t.requires_grad_(True)
    
    # Generate random data and smooth it to ensure continuity
    random_data = torch.randn(n_samples).view(-1, 1)
    smoothed_random_data = smooth_data(random_data, window_size).view(-1, 1)
    
    # Interpolating smoothed data to ensure connection to t
    t_normalized = (t - t.min()) / (t.max() - t.min())  # Normalize t to [0, 1]
    x = F.interpolate(smoothed_random_data.view(1, 1, -1), size=n_samples, mode='linear', align_corners=True).view(-1, 1)
    x.requires_grad_(True)
    
    g_x = torch.full_like(t, g_x_true)
    
    alpha_x_times_x = alpha_x_true * x
    d_alpha_x_dt = torch.autograd.grad(outputs=alpha_x_times_x, inputs=t,
                                       grad_outputs=torch.ones_like(alpha_x_times_x),
                                       create_graph=True, retain_graph=True)[0]
    
    d2_alpha_x_dt2 = torch.autograd.grad(outputs=d_alpha_x_dt, inputs=t,
                                         grad_outputs=torch.ones_like(d_alpha_x_dt),
                                         create_graph=True)[0]
    
    a_x_obs = d2_alpha_x_dt2 + beta_x_true * g_x
    
    return t, x, g_x, a_x_obs

# Example usage
n_samples = 100
t, x, g_x, a_x_obs = generate_data(n_samples)
print(f"t shape: {t.shape}")
print(f"x shape: {x.shape}")
print(f"g_x shape: {g_x.shape}")
print(f"a_x_obs shape: {a_x_obs.shape}")
