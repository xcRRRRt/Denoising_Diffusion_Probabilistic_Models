# Denoising Diffusion Probabilistic Models

---

## Usage

```python
from module.unet import UNet
from module.ddpm import DDPM
from trainer import train
from util.data import fashion_mnist_data_loader

if __name__ == '__main__':
    model = UNet(
        in_feature=1,
        num_features=64,
        multi=[1, 2, 4],
        shape=(1, 28, 28)
    )
    ddpm = DDPM(
        model,
        T=1000,
        d=100,
        beta=(1e-4, 0.02)
    )
    train_loader = fashion_mnist_data_loader("./dataset/", batch_size=64, train=True, num_workers=4)
    test_loader = fashion_mnist_data_loader("./dataset/", batch_size=64, train=False, num_workers=4)
    train(ddpm, train_loader, test_loader, epochs=100)
```

### Monitor Training

`tensorboard --logdir=./lightning_logs`

---

## Result

### 1. Fashion Mnist

(about 30 minutes)

<img src="readme/fashion_mnist.png" width="224" alt="Fashion MNIST">

### 2. CIFAR10

---

## Experiment params

|    dataset    |  img size   | num features |   multi   | Params |
|:-------------:|:-----------:|:------------:|:---------:|:------:|
| Fashion MNIST | (1, 28, 28) |      64      | [1, 2, 4] |        |
|    CIFAR10    | (3, 32, 32) |              |           |        |

