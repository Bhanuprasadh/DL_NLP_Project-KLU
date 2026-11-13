from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def load_mnist():
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    return train_dataset, test_dataset


def main():
    train_dataset, test_dataset = load_mnist()

    print("Training samples:", len(train_dataset))
    print("Test samples:", len(test_dataset))

    image, label = train_dataset[0]
    print("Image size:", image.shape)
    print("Actual label:", label)
    visualize_mnist(train_dataset)


def visualize_mnist(dataset, num_images=10):
    plt.figure(figsize=(12, 5))

    for i in range(num_images):
        image, label = dataset[i]
        plt.subplot(2, 5, i+1)
        plt.imshow(image.squeeze(), cmap='gray')
        plt.title(f"Actual Label :{label}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()