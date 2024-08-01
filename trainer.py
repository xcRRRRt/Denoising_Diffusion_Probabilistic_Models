import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as f
import torchvision
from matplotlib.backends.backend_agg import FigureCanvasAgg

from module.ddpm import DDPM

torch.set_float32_matmul_precision("high")


class Diffusion(pl.LightningModule):
    def __init__(self, ddpm: DDPM):
        super().__init__()
        self.ddpm = ddpm

    def training_step(self, batch, batch_idx):
        x_0 = batch[0]
        _, eps, eps_theta = self.ddpm(x_0)
        loss = f.mse_loss(eps, eps_theta)
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x_0 = batch[0]
        _, eps, eps_theta = self.ddpm(x_0)
        loss = f.mse_loss(eps, eps_theta)
        self.log('valid/loss', loss)
        if batch_idx == 0:
            c = x_0.shape[1]

            sample = self.ddpm.sample(32, x_0.shape[1:], self.device)
            if c == 1:
                grid = self.show_images(sample)
            else:
                grid = torchvision.utils.make_grid(sample)

            self.logger.experiment.add_image('sample', grid, global_step=self.global_step, dataformats="CHW")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
        return [optimizer], [scheduler]

    def show_images(self, images):
        # Converting images to CPU numpy arrays
        if isinstance(images, torch.Tensor):
            images = images.detach().cpu().numpy()

        # Defining number of rows and columns
        rows = int(len(images) ** (1 / 2))
        cols = (len(images) + rows - 1) // rows  # 确保能容纳所有图像
        fig = plt.figure()

        # Populating figure with sub-plots
        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx < len(images):
                    ax = fig.add_subplot(rows, cols, idx + 1)
                    ax.imshow(images[idx][0], cmap="gray")
                    ax.axis("off")  # 隐藏轴
                    idx += 1

        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)  # 减少子图之间的间距

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        matrix = np.asarray(buf)[:, :, :3].transpose((2, 0, 1))
        plt.close('all')
        return torch.from_numpy(matrix)


def train(ddpm: DDPM, train_dataloader, val_dataloader, epochs: int, checkpoint_path: str = None):
    diffusion = Diffusion(ddpm)
    trainer = pl.Trainer(
        log_every_n_steps=1,
        accelerator='gpu',
        max_epochs=epochs,
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor='valid/loss', save_top_k=1, verbose=True, mode='min')
        ]
    )
    trainer.fit(diffusion, train_dataloader, val_dataloader, ckpt_path=checkpoint_path)
