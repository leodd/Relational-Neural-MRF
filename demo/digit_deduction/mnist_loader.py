import numpy as np
import torchvision
from matplotlib import pyplot as plt


class MNISTRandomDigit():
    def __init__(self, root='.'):
        self.data = torchvision.datasets.MNIST(root, train=True, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                 (0.1307,), (0.3081,))
            ])
        )

        self.digit_map = [list() for _ in range(10)]

        for idx, digit in enumerate(self.data.targets):
            self.digit_map[digit].append(idx)

    def get_random_image(self, digit):
        idx = np.random.choice(self.digit_map[digit])
        img = self.data[idx][0]
        return img[0].numpy()


if __name__ == '__main__':
    dataset = MNISTRandomDigit()
    img = dataset.get_random_image(0)
    plt.imshow(img)
    plt.show()
