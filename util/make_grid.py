import torch
import torchvision
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np


def make_grid(tensor4d: torch.Tensor):
    c = tensor4d.shape[1]
    if c == 1:
        grid = make_grid_gray(tensor4d)
    elif c == 3:
        grid = make_grid_rgb(tensor4d)
    else:
        raise ValueError("Unsupported image channels, should be 1 or 3")
    return grid


def make_grid_rgb(tensor4d):
    return torchvision.utils.make_grid(tensor4d)


def make_grid_gray(tensor4d):
    # Converting images to CPU numpy arrays
    if isinstance(tensor4d, torch.Tensor):
        tensor4d = tensor4d.detach().cpu().numpy()

    # Defining number of rows and columns
    rows = int(len(tensor4d) ** (1 / 2))
    cols = (len(tensor4d) + rows - 1) // rows  # 确保能容纳所有图像
    fig = plt.figure()

    # Populating figure with sub-plots
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx < len(tensor4d):
                ax = fig.add_subplot(rows, cols, idx + 1)
                ax.imshow(tensor4d[idx][0], cmap="gray")
                ax.axis("off")  # 隐藏轴
                idx += 1

    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)  # 减少子图之间的间距

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    matrix = np.asarray(buf)[:, :, :3].transpose((2, 0, 1))
    plt.close('all')
    return torch.from_numpy(matrix)
