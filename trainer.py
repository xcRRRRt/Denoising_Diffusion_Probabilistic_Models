import sys

import lightning.pytorch as pl
import torch
import torch.nn.functional as f
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm

from module.ddpm import DDPM
from util.make_grid import make_grid

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
            sample = self.ddpm.sample(32, x_0.shape[1:], self.device)
            self.logger.experiment.add_image('sample', make_grid(sample), global_step=self.global_step, dataformats="CHW")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
        return [optimizer], [scheduler]


class ProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self) -> Tqdm:
        return Tqdm(
            desc=self.validation_description,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
            bar_format=self.BAR_FORMAT,
        )


def train(ddpm: DDPM, train_dataloader, val_dataloader, epochs: int, checkpoint_path: str = None):
    diffusion = Diffusion(ddpm)
    trainer = pl.Trainer(
        log_every_n_steps=1,
        accelerator='gpu',
        max_epochs=epochs,
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor='valid/loss', save_top_k=1, verbose=True, mode='min'),
            ProgressBar()
        ]
    )
    trainer.fit(diffusion, train_dataloader, val_dataloader, ckpt_path=checkpoint_path)
