'''
    This file is a implementation of Time Series Wrapper.
    Specifically, it samples N frames from a video with N frames according to a truncated normal distribution. Therefore, it can be regarded as local acceleration or deceleration within a video.
    It can serve as a strong data augmentation for action segnemtation task by setting the 'if_warp=True' in batch_gen.BatchGenerator.next_batch. We do not use this trick in our paper, but it does give better results :).
'''
import numpy as np
from scipy.stats import truncnorm
import torch.nn.functional as TF
import torch.nn as nn

class TimeWarpLayer(nn.Module):
    def __init__(self):
        super(TimeWarpLayer, self).__init__()

    def forward(self, x, grid, mode='bilinear'):
        '''
        :type&shape x: (cuda.)FloatTensor, (N, D, T)
        :type&shape grid: (cuda.)FloatTensor, (N, T, 2)
        :type&mode: bilinear or nearest
        :rtype&shape: (cuda.)FloatTensor, (N, D, T)
        '''
        assert len(x.shape) == 3
        assert len(grid.shape) == 3
        assert grid.shape[-1] == 2
        x_4dviews = list(x.shape[:2]) + [1] + list(x.shape[2:])
        grid_4dviews = list(grid.shape[:1]) + [1] + list(grid.shape[1:])
        out = TF.grid_sample(input=x.view(x_4dviews), grid=grid.view(grid_4dviews), mode=mode, align_corners=True).view(x.shape)
        return out


class GridSampler():
    def __init__(self, N_grid, low=1, high=5):  # high=5
        N_primary = 100 * N_grid
        assert N_primary % N_grid == 0
        self.N_grid = N_grid
        self.N_primary = N_primary
        self.low = low
        self.high = high

    def sample(self, batchsize=1):
        num_centers = np.random.randint(low=self.low, high=self.high)
        lower, upper = 0, 1
        mu, sigma = np.random.rand(num_centers), 1 / (num_centers * 1.5)  # * 1.5
        TN = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        vals = TN.rvs(size=(self.N_primary, num_centers))
        grid = np.sort(
            np.random.choice(vals.reshape(-1), size=self.N_primary, replace=False))  # pick one center for each primary
        grid = (grid[::int(self.N_primary / self.N_grid)] * 2 - 1).reshape(1, self.N_grid, 1)  # range [-1, 1)
        grid = np.tile(grid, (batchsize, 1, 1))
        grid = np.concatenate([grid, np.zeros_like(grid)], axis=-1)
        return grid


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    grid_sampler = GridSampler(N_grid=1000)
    grid = grid_sampler.sample(1)
    assert np.all(grid[:, :, 1] == 0)
    assert np.all(grid[0, ...] == grid[-1, ...])
    print(np.min(grid), np.max(grid))
    plt.hist(grid[0, :, 0], bins=50)
    plt.show()