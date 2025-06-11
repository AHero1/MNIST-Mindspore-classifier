import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import Tensor, context
import numpy as np
from mindspore import Model

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

def load_csv(path):
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    images = data[:, 1:].reshape((-1, 1, 28, 28)).astype(np.float32) / 255.0
    labels = data[:, 0].astype(np.int32)
    return images, labels

def create_dataset(images, labels, batch_size=32):
    dataset = ds.NumpySlicesDataset({"image": images, "label": labels}, shuffle=True)
    dataset = dataset.batch(batch_size)
    return dataset

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.SequentialCell([
            nn.Dense(28*28, 128),
            nn.ReLU(),
            nn.Dense(128, 10)
        ])
        
    def construct(self, x):
        x = self.flatten(x)
        return self.fc(x)

def train():
    train_images, train_labels = load_csv("/Users/ahero1/Downloads/mnist_csv/archive/mnist_train.csv")
    test_images, test_labels = load_csv("/Users/ahero1/Downloads/mnist_csv/archive/mnist_test.csv")

    train_ds = create_dataset(train_images, train_labels)
    test_ds = create_dataset(test_images, test_labels)

    network = Net()
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    optimizer = nn.Adam(network.trainable_params(), learning_rate=0.001)

    model = Model(network, loss_fn, optimizer, metrics={"Accuracy": nn.Accuracy()})
    model.train(5, train_ds, dataset_sink_mode=False)
    acc = model.eval(test_ds, dataset_sink_mode=False)
    print("Test Accuracy:", acc)

if __name__ == "__main__":
    train()

