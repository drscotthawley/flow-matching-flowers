
import torch 
import numpy as np
from scipy.stats import wasserstein_distance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

@torch.no_grad()
def jensen_shannon_divergence(p, q, eps=1e-10, device='cuda'):
    """
    Compute Jensen-Shannon divergence between two sample sets.
    Fully torch-based and GPU compatible.
    """
    # Get range for histograms
    min_val = min(p.min(), q.min())
    max_val = max(p.max(), q.max())
    bins = min(100, len(p) // 10)  # Rule of thumb for bin count

    # Compute histograms
    p_hist = torch.histc(p, bins=bins, min=min_val, max=max_val).to(device)
    q_hist = torch.histc(q, bins=bins, min=min_val, max=max_val).to(device)

    # Normalize to probability distributions
    p_hist = p_hist / p_hist.sum()
    q_hist = q_hist / q_hist.sum()

    # Add small epsilon to avoid log(0)
    p_hist = p_hist + eps
    q_hist = q_hist + eps

    # Renormalize after adding epsilon
    p_hist = p_hist / p_hist.sum()
    q_hist = q_hist / q_hist.sum()

    # Compute JS divergence
    m = (p_hist + q_hist) * 0.5
    js_div = 0.5 * (torch.sum(p_hist * torch.log(p_hist / m)) +
                    torch.sum(q_hist * torch.log(q_hist / m)))

    return js_div.item()  # Convert to Python scalar


def wasserstein_metric(p, q):
    """
    Compute the 1D Wasserstein distance between two sample sets.
    """
    return wasserstein_distance(p.cpu().numpy(), q.cpu().numpy())

@torch.no_grad()
def mmd_rbf(x, y, sigma=None,):
    """
    Fixed version of MMD calculation
    """
    #x = torch.tensor(x, dtype=torch.float32, device=x.device)  # Ensure float type
    #y = torch.tensor(y, dtype=torch.float32)
    device = x.device

    # Reshape to 2D if needed
    if x.dim() == 1:
        x = x.view(-1, 1)
    if y.dim() == 1:
        y = y.view(-1, 1)

    # Compute squared distances
    xx = torch.cdist(x, x, p=2).pow(2).to(device)
    yy = torch.cdist(y, y, p=2).pow(2).to(device)
    xy = torch.cdist(x, y, p=2).pow(2).to(device)

    if sigma is None:
        # More robust sigma calculation
        sigma = torch.median(torch.sqrt(torch.cat([xx.view(-1), yy.view(-1), xy.view(-1)])))
        sigma = torch.clamp(sigma, min=1e-6)  # Prevent zero sigma

    # Compute kernel values
    k_xx = torch.exp(-xx / (2 * sigma**2)).mean()
    k_yy = torch.exp(-yy / (2 * sigma**2)).mean()
    k_xy = torch.exp(-xy / (2 * sigma**2)).mean()

    return float(k_xx + k_yy - 2 * k_xy)  # Ensure return scalar

def compare_distributions(target, generated):
    """
    Compare two sample sets using multiple metrics.
    """
    # Convert to numpy if needed


    # Flatten if multi-dimensional
    target = target.reshape(-1)
    generated = generated.reshape(-1)

    metrics = {
        'J-S': jensen_shannon_divergence(target, generated),
        'Wass': wasserstein_metric(target, generated), # note that this is CPU-only but still isn't too slow
        'mmd_rbf': mmd_rbf(target.reshape(-1, 1), generated.reshape(-1, 1)),
        # # these relative errors can appear large when target values are small; thus they can be misleanding -- e.g. the mean when it's near zero
        # '%d_min':  torch.abs((target.min() - generated.min()) /target.min()).item(),
        # '%d_max':  torch.abs((target.max() - generated.max()) /target.max()).item(),
        # '%d_mean': torch.abs((target.mean()- generated.mean())/target.mean()).item(),
        # '%d_std':  torch.abs((target.std() - generated.std()) /target.std()).item(),
        # absolute errors will not vary based on target values, though the first 3 I don't find all that helpful
        # 'd_min':  torch.abs(target.min() - generated.min()).item(),
        # 'd_max':  torch.abs(target.max() - generated.max()).item(),
        # 'd_mean': torch.abs(target.mean()- generated.mean()).item(),
        'd_std':  torch.abs(target.std() - generated.std()).item(),
    }

    return metrics

def make_histograms(dist_list, colors=['red','green','blue'], labels=['out','gen','init'], bins=32,filename="hist.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, dist in enumerate(dist_list):
      counts, bins = np.histogram(dist.cpu().numpy().flatten(), bins=bins)
      ax.stairs(counts, bins, alpha=0.5, color=colors[i], label=labels[i], fill=True)
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(filename)
    plt.close()
    return filename


if __name__ == '__main__':
  # just some testing code 

  def create_uniform_dist(n_points=1000, n_dims=1, min_val=-1.0, max_val=1.0):
      # Generate uniformly distributed points in the specified number of dimensions
      # Author: Raymond Fan
      return torch.rand(n_points, n_dims) * (max_val - min_val) + min_val
  
  def create_gaussian_dist(n_points=1000, n_dims=1, mean=0.0, std_dev=1/3.0):
      # Generate Gaussian-distributed points
      # Why std 1/3? 1/3 is the variance of a uniform distribution from -1 to 1
      # Author: Raymond Fan
      return torch.randn(n_points, n_dims) * std_dev + mean

  device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
  print("device =",device)
  # test the metrics
  batch_size = 512
  dist1 = create_gaussian_dist(n_points=batch_size).to(device)
  dist2 = create_uniform_dist(n_points=batch_size).to(device)
  dist2_again = create_uniform_dist(n_points=batch_size).to(device)
  print("dist1 vs dist1:         ",compare_distributions(dist1, dist1)) # should be all zeros
  print("dist2 vs dist2:         ",compare_distributions(dist2, dist2)) # should be all zeros
  print("dist1 vs dist2:         ",compare_distributions(dist1, dist2)) # asymptotic value of gen vs init should approach these values
  print("dist1 vs dist2:         ",compare_distributions(dist1, dist2)) # asymptotic value of gen vs init should approach these values
  print("dist2 vs dist2_again:   ",compare_distributions(dist2, dist2_again)) # "best you can hope for", given batch_size
  print("making histograms")
  out_file = make_histograms([dist1, dist2])
