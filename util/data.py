import torch
import torchvision
from torch.utils.data import DataLoader


def _transform_function(x):
    return (x - 0.5) * 2


_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(_transform_function),
])


def cifar10_data_loader(root, batch_size, train: bool, num_workers: int):
    data = torchvision.datasets.CIFAR10(
        root=root,
        train=train,
        transform=_transform,
        download=True
    )
    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )
    return loader


def mnist_data_loader(root, batch_size, train: bool, num_workers: int):
    data = torchvision.datasets.MNIST(
        root=root,
        train=train,
        transform=_transform,
        download=True
    )
    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )
    return loader


def celeba_data_loader(root, batch_size, train: bool, num_workers: int):
    data = torchvision.datasets.CelebA(
        root=root,
        split='train' if train else 'valid',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(64),
            torchvision.transforms.ToTensor()
        ]),
        download=True
    )
    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0)
    )
    return loader


def fashion_mnist_data_loader(root, batch_size, train: bool, num_workers: int):
    data = torchvision.datasets.FashionMNIST(
        root=root,
        train=train,
        transform=_transform,
        download=True
    )
    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )
    return loader


def _show_images(images):
    import matplotlib.pyplot as plt
    # Converting images to CPU numpy arrays
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()

    # Defining number of rows and columns
    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

    # Populating figure with sub-plots
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                plt.imshow(images[idx][0], cmap="gray")
                idx += 1
    # Showing the figure
    plt.show()


if __name__ == '__main__':
    loader_ = fashion_mnist_data_loader('../dataset/', train=False, batch_size=32, num_workers=4)
    batch = next(iter(loader_))[0]
    _show_images(batch)
