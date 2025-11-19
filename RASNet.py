import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
image_size = 64  # Resize images to 64x64
batch_size = 128
latent_dim = 100  # Dimension of the latent vector (z)
learning_rate = 0.0002
num_epochs = 50
beta1 = 0.5  # Adam optimizer beta1 hyperparameter

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),  # Make sure images are square
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

# Example: Using CIFAR-10 dataset
dataset = torchvision.datasets.CIFAR10(root='./Dataset', train=True,
                                        download=True, transform=transform)

# Alternatively, use ImageFolder for custom datasets:
# dataset = torchvision.datasets.ImageFolder(root='./path/to/your/dataset', transform=transform)


dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=2)  # Adjust num_workers based on your system

# --- ResNet-based Discriminator ---
class ResNetDiscriminator(nn.Module):
    def __init__(self, num_channels=3, ndf=64):  # ndf: discriminator feature map size
        super(ResNetDiscriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (num_channels) x 64 x 64
            nn.Conv2d(num_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# --- Generator Model ---
class Generator(nn.Module):
    def __init__(self, latent_dim, num_channels=3, ngf=64): # ngf: generator feature map size
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # output size. (num_channels) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)



# --- Model instantiation ---
generator = Generator(latent_dim).to(device)
discriminator = ResNetDiscriminator().to(device)

# --- Loss function and optimizers ---
criterion = nn.BCELoss()  # Binary Cross Entropy Loss

optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))

# --- Helper function for displaying generated images ---
def show_images(images):
    images = images.cpu().detach()  # Move to CPU and detach from computation graph
    images = images * 0.5 + 0.5  # Unnormalize
    grid = make_grid(images)
    plt.imshow(grid.permute(1, 2, 0))  # Transpose for matplotlib
    plt.axis('off')
    plt.show()


# --- Training loop ---
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        # 1. Train Discriminator
        #   - Train with real images
        #   - Train with fake images

        # Real images
        real_labels = torch.ones(batch_size).to(device)  # All real images have label 1
        outputs = discriminator(real_images).view(-1) # reshape to 1D
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs.mean().item()

        # Fake images
        z = torch.randn(batch_size, latent_dim, 1, 1).to(device) # Latent vector
        fake_images = generator(z)
        fake_labels = torch.zeros(batch_size).to(device) # All fake images have label 0
        outputs = discriminator(fake_images.detach()).view(-1) # detach to avoid generator training
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs.mean().item()

        # Backprop and optimize discriminator
        d_loss = d_loss_real + d_loss_fake
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()


        # 2. Train Generator
        #   - Train to fool the discriminator

        z = torch.randn(batch_size, latent_dim, 1, 1).to(device)
        fake_images = generator(z)
        outputs = discriminator(fake_images).view(-1)  # No detach here - generator needs to train!
        real_labels = torch.ones(batch_size).to(device) #  Trick discriminator into thinking they are real
        g_loss = criterion(outputs, real_labels)

        # Backprop and optimize generator
        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'.format(
                epoch+1, num_epochs, i+1, len(dataloader), d_loss.item(), g_loss.item(), real_score, fake_score))

    # ---  Example of displaying generated images after each epoch ---
    with torch.no_grad():
        z = torch.randn(64, latent_dim, 1, 1).to(device)
        generated_images = generator(z)
        show_images(generated_images)

print("Training finished!")

# --- Save the models ---
torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')