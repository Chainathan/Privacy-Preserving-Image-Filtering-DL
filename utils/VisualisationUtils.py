import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def show_images(images, titles=None, save_path=None, figsize=(10, 10)):
    """
    Display images in a grid
    :param images: list of images
    :param titles: list of titles
    :param save_path: path to save the image
    :return:
    """
    plt.figure(figsize=figsize)
    if not isinstance(images, (list, np.ndarray)):
        plt.imshow(images)
        plt.axis('off')
        plt.show()
        if save_path is not None:
            plt.savefig(save_path)
        return
    n_images = len(images)
    if titles is None:
        titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig, axes = plt.subplots(1, n_images, figsize=figsize)
    for n, (image, title) in enumerate(zip(images, titles)):
        axes[n].imshow(image)
        axes[n].set_title(title)
        axes[n].axis('off')
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()