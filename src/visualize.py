import matplotlib.pyplot as plt

def show_images(images, title):

    images = images[:10]

    plt.figure(figsize=(10,2))

    for i in range(10):
        plt.subplot(1,10,i+1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.axis('off')

    plt.suptitle(title)
    plt.show()

import matplotlib.pyplot as plt
import torch
from src.preprocess import flatten

def show_reconstruction(model, data_loader):

    model.eval()

    with torch.no_grad():

        images, _ = next(iter(data_loader))

        # flatten
        inputs = flatten(images)

        # model output
        outputs = model(inputs)

        # reshape back to image
        outputs = outputs.view(-1, 1, 28, 28)

        # plot
        plt.figure(figsize=(10,4))

        for i in range(10):
            # original
            plt.subplot(2,10,i+1)
            plt.imshow(images[i].squeeze(), cmap='gray')
            plt.axis('off')

            # reconstructed
            plt.subplot(2,10,i+11)
            plt.imshow(outputs[i].squeeze(), cmap='gray')
            plt.axis('off')

        plt.suptitle("Top: Original | Bottom: Reconstructed")
        plt.show()

def show_denoising(model, data_loader):

    import torch
    import matplotlib.pyplot as plt
    from src.preprocess import add_noise

    model.eval()

    with torch.no_grad():

        images, _ = next(iter(data_loader))

        noisy = add_noise(images)

        outputs = model(noisy)

        plt.figure(figsize=(10,6))

        for i in range(10):

            # noisy
            plt.subplot(3,10,i+1)
            plt.imshow(noisy[i].squeeze(), cmap='gray')
            plt.axis('off')

            # reconstructed
            plt.subplot(3,10,i+11)
            plt.imshow(outputs[i].squeeze(), cmap='gray')
            plt.axis('off')

            # original
            plt.subplot(3,10,i+21)
            plt.imshow(images[i].squeeze(), cmap='gray')
            plt.axis('off')

        plt.suptitle("Top: Noisy | Middle: Denoised | Bottom: Original")
        plt.show()