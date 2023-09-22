import os
import time

import matplotlib.pyplot as plt  # plotting library
import numpy as np  # this module is useful to work with numerical arrays
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
from torchvision.datasets import MNIST


class Encoder(nn.Module):

    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()

        # Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=2),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=2),
            nn.ReLU(True),
        )

        # Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        # Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(3*3*128, fc2_input_dim),
            nn.ReLU(True),
            nn.Linear(fc2_input_dim, encoded_space_dim)
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class Decoder(nn.Module):

    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, fc2_input_dim),
            nn.ReLU(True),
            nn.Linear(fc2_input_dim, 128*3*3),
            nn.ReLU(True)
        )

        self.unFlatten = nn.Unflatten(dim=1,
                                      unflattened_size=(128, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=2, output_padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=0, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unFlatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x


# Training function
def train_epoch(f_encoder, f_decoder, f_device, dataloader, f_lossFunction, f_optimizer):
    # Set train mode for both the encoder and the decoder
    f_encoder.train()
    f_decoder.train()
    training_loss = []
    counter = 0
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for image_batch, _ in dataloader:
        # with "_" we just ignore the labels (the second element of the dataloader tuple)
        # Move tensor to the proper device
        image_batch = image_batch.to(f_device)
        # Encode data
        encoded_data = f_encoder(image_batch)
        # Decode data
        decoded_data = f_decoder(encoded_data)
        # Evaluate loss
        loss = f_lossFunction(decoded_data, image_batch)
        # Backward pass
        f_optimizer.zero_grad()
        loss.backward()
        f_optimizer.step()
        # Print batch loss
        counter += dataloader.batch_size
        if counter >= 12000:
            print('\tMini-batch loss: %f' % loss.data)
            counter -= 12000
        training_loss.append(loss.detach().cpu().numpy())

    return np.mean(training_loss)


# Testing function
def test_epoch(f_encoder, f_decoder, f_device, dataloader, loss_fn):
    # Set evaluation mode for encoder and decoder
    f_encoder.eval()
    f_decoder.eval()
    # Define the lists to store the outputs for each batch
    concat_outputs = []
    concat_labels = []
    with torch.no_grad():  # No need to track the gradients
        for image_batch, _ in dataloader:
            # Move tensor to the proper device
            image_batch = image_batch.to(f_device)
            # Encode data
            encoded_data = f_encoder(image_batch)
            # Decode data
            decoded_data = f_decoder(encoded_data)
            # Append the network output and the original image to the lists
            concat_outputs.append(decoded_data.cpu())
            concat_labels.append(image_batch.cpu())
        # Create a single tensor with all the values in the lists
        concat_outputs = torch.cat(concat_outputs)
        concat_labels = torch.cat(concat_labels)
        # Evaluate global loss
        dataset_loss = loss_fn(concat_outputs, concat_labels)
    return dataset_loss.data


def plot_ae_outputs(f_encoder, f_decoder, n=10):
    plt.figure(figsize=(16, 4.5))
    targets = dataset_test.targets.numpy()
    t_idx = {i: np.where(targets == i)[0][0] for i in range(n)}
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        img = dataset_test[t_idx[i]][0].unsqueeze(0).to(device)
        f_encoder.eval()
        f_decoder.eval()
        with torch.no_grad():
            rec_img = f_decoder(f_encoder(img))
        plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title('Original images')
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title('Reconstructed images')
    plt.show()


def plot_pca_outputs(f_device, n=10):
    plt.figure(figsize=(16, 4.5))
    targets = dataset_test.targets.numpy()
    t_idx = {i: np.where(targets == i)[0][0] for i in range(n)}

    concat_labels = []
    for image_batch, _ in trainLoader:
        image_batch = image_batch.to(f_device)
        concat_labels.append(image_batch.cpu())
    concat_labels = torch.cat(concat_labels)

    pca = PCA(n_components=0.6)
    pca.fit(concat_labels.numpy().reshape((60000, 784)))
    print('PCA is using', pca.transform([concat_labels.numpy().reshape((60000, 784))[0]]).shape[1], 'and AE', d)

    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        img = dataset_test[t_idx[i]][0].unsqueeze(0).to(device)
        rec_img = torch.from_numpy(pca.inverse_transform(pca.transform(img.reshape(1, 784))).reshape((1, 28, 28)))
        plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title('Original images')
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title('Reconstructed PCA images')
    plt.show()


def pca_loss(f_device, loss_fn, loader, components):
    concat_labels = []
    for image_batch, _ in trainLoader:
        image_batch = image_batch.to(f_device)
        concat_labels.append(image_batch.cpu())
    concat_labels = torch.cat(concat_labels)
    data = concat_labels.numpy().reshape((60000, 784))
    pca = PCA(n_components=components)
    pca.fit(data)

    concat_labels = []
    for image_batch, _ in loader:
        image_batch = image_batch.to(f_device)
        concat_labels.append(image_batch.cpu())
    concat_labels = torch.cat(concat_labels)
    data = concat_labels.numpy().reshape((10000, 784))
    # Encode data
    encoded_data = pca.transform(data)
    # Decode data
    decoded_data = pca.inverse_transform(encoded_data)
    reshaped_data = decoded_data.reshape((10000, 1, 28, 28))
    tensor_data = torch.from_numpy(reshaped_data)
    # Evaluate global loss
    dataset_loss = loss_fn(tensor_data, concat_labels)
    return dataset_loss.data


if __name__ == "__main__":
    dataset = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
    dataset_test = MNIST(os.getcwd(), train=False, download=False, transform=transforms.ToTensor())
    dataset_validation = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())

    _, dataset_validation.data, _, dataset_validation.targets = train_test_split(dataset_validation.data,
                                                                                 dataset_validation.targets,
                                                                                 test_size=1 / 6)

    batch_size = 256
    device = torch.device("cpu")

    trainLoader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    testLoader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=1)
    valLoader = torch.utils.data.DataLoader(dataset_validation, batch_size=batch_size, shuffle=True, num_workers=1)

    # Define the loss function
    loss_function = torch.nn.SmoothL1Loss()

    # Define an optimizer (both for the encoder and the decoder!)
    lr = 0.001

    # Set the random seed for reproducible results
    torch.manual_seed(0)

    # Initialize the two networks
    d = 4

    # model = AutoEncoder(encoded_space_dim=encoded_space_dim)
    encoder = Encoder(encoded_space_dim=d, fc2_input_dim=128*4)
    decoder = Decoder(encoded_space_dim=d, fc2_input_dim=128*4)
    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ]

    optimizer = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

    encoder.to(device)
    decoder.to(device)

    num_epochs = 50
    array_epoch = []
    array_info = np.zeros((num_epochs, 4))

    pca_loss_test = pca_loss(device, loss_function, testLoader, 0.6)
    pca_loss_val = pca_loss(device, loss_function, valLoader, 0.6)
    plot_pca_outputs(device)
    print('-----------------------------------------------------')
    for epoch in range(num_epochs):
        print('Starting training epoch', epoch + 1, ':')
        start = time.time()
        train_loss = train_epoch(encoder, decoder, device, trainLoader, loss_function, optimizer)
        end = time.time()
        print(f'Training Loss:', train_loss)
        print('Training time needed:', end - start)
        val_loss = test_epoch(encoder, decoder, device, valLoader, loss_function)
        print(f'Validation Loss: %f' % val_loss)
        test_loss = test_epoch(encoder, decoder, device, testLoader, loss_function)
        print(f'Test Loss: %f' % test_loss)
        if test_loss <= pca_loss_test:
            print('AE Test loss was smaller than PCA loss by: %f' % (pca_loss_test - test_loss))
        else:
            print('PCA loss was smaller than AE test loss by: %f' % (test_loss - pca_loss_test))

        if val_loss <= pca_loss_val:
            print('AE Validation loss was smaller than PCA loss by: %f' % (pca_loss_test - val_loss))
        else:
            print('PCA loss was smaller than AE validation loss by: %f' % (val_loss - pca_loss_test))
        if epoch % 10 == 9:
            plot_ae_outputs(encoder, decoder, n=10)
        array_info[epoch][0] = train_loss
        array_info[epoch][1] = val_loss
        array_info[epoch][2] = test_loss
        array_info[epoch][3] = end - start
        array_epoch.append(f'Epoch {epoch + 1}')
        print('-----------------------------------------------------')
    df = pd.DataFrame(array_info,
                      index=array_epoch, columns=['Training Loss', 'Validation Loss', 'Testing Loss', 'Time'])
    df.to_excel('5-layered-lr=0.001-batch=256-loss_function=SmoothL1-ReLU.xlsx', sheet_name='sheet_one')
