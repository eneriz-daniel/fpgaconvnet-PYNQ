import numpy as np

# Grabs image at index idx from the MNIST testing dataset
def get_MNIST_image(idx):
    # Read the mnist binary format
    image_path = "assets/MNIST/t10k-images-idx3-ubyte"
    with open(image_path, 'rb') as fh:
        fh.seek(16 + idx * (28 * 28), 0)
        arr = list(fh.read(28 * 28))
    arr = np.array(arr)
    arr = arr.astype('float32')

    # Normalize array [-1, 1]
    arr = np.subtract(arr, (np.amax(arr) + np.amin(arr)) / 2)
    arr = np.multiply(arr, 1 / np.amax(arr))

    arr = np.reshape(arr, (28, 28))
    arr.transpose()

    # Batches for onnx run
    arr = np.reshape(arr, (1, 1, 28, 28))
    arr = np.tile(arr, (42, 1, 1, 1))
    return arr


# Grabs corresponding label at index idx from the MNIST testing dataset
def get_MNIST_label(idx):
    label_path = "assets/MNIST/t10k-labels-idx1-ubyte"
    with open(label_path, 'rb') as fh:
        fh.seek(8 + idx, 0)
        return int(fh.read(1)[0])
    