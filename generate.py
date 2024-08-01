import torch
import torchvision

from module.ddpm import DDPM
from module.unet import UNet
from trainer import Diffusion

if __name__ == '__main__':
    model = UNet(3, 128, [1, 2, 4, 8], (3, 64, 64))
    ddpm = DDPM(model, T=1000, d=100)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    diffusion: Diffusion = Diffusion.load_from_checkpoint(r'lightning_logs/version_4/checkpoints/epoch=9-step=50870.ckpt', ddpm=ddpm)
    tensor4d = diffusion.ddpm.sample(32, shape=(3, 64, 64), device=device)
    grid = torchvision.utils.make_grid(tensor4d)
    torchvision.utils.save_image(grid, r'readme/celeba.png')
