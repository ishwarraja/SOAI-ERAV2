import matplotlib.pyplot as plt
import numpy as np

def display_random_images(trainloader):
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # Display 10 random images from the batch
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(np.transpose(images[i], (1, 2, 0)))
        ax.axis('off')
        ax.set_title(f'Label: {labels[i].item()}')
    plt.show()
