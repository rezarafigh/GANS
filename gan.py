import torch
from torch.optim import Adam
from torch import nn
from generator import Generator
from discriminator import Discriminator
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Load data
def get_data_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root='./data', download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    return dataloader

# Training function
def train_gan(dataloader, device):
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    g_optimizer = Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    num_epochs = 50

    for epoch in range(num_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            real_imgs = imgs.to(device)
            real_labels = torch.ones(imgs.size(0), 1).to(device)
            fake_labels = torch.zeros(imgs.size(0), 1).to(device)

            # Train Discriminator
            d_optimizer.zero_grad()
            outputs = discriminator(real_imgs)
            d_real_loss = criterion(outputs, real_labels)
            z = torch.randn(imgs.size(0), 100).to(device)
            fake_imgs = generator(z)
            outputs = discriminator(fake_imgs.detach())
            d_fake_loss = criterion(outputs, fake_labels)
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            g_optimizer.zero_grad()
            outputs = discriminator(fake_imgs)
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            g_optimizer.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch}/{num_epochs}] Batch {i}/{len(dataloader)} \
                      Loss D: {d_loss.item()}, loss G: {g_loss.item()}")

if __name__ == "__main__":
    dataloader = get_data_loader()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_gan(dataloader, device)
