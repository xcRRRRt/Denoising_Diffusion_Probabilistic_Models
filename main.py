from module.ddpm import DDPM
from module.unet import UNet
from trainer import train
from util.data import celeba_data_loader

if __name__ == '__main__':
    model = UNet(3, 128, [1, 2, 4, 8], (3, 64, 64))
    ddpm = DDPM(model, T=1000, d=100)
    train_loader = celeba_data_loader(r"E:\Dataset", batch_size=32, train=True, num_workers=4)
    test_loader = celeba_data_loader(r"E:\Dataset", batch_size=32, train=False, num_workers=4)
    train(ddpm, train_loader, test_loader, epochs=1000)
