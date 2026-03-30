"""
GAN Mode Collapse Training Script
Trains both DCGAN and WGAN-GP models and saves results
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Dataset
# ============================================================================

class ImageDataset(Dataset):
    """Custom dataset for loading images from directory"""
    def __init__(self, root_dir, transform=None, max_samples=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_files.append(os.path.join(root, file))
        
        if max_samples and len(self.image_files) > max_samples:
            self.image_files = self.image_files[:max_samples]
        
        print(f"Found {len(self.image_files)} images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


# ============================================================================
# Model Architectures
# ============================================================================

class DCGANGenerator(nn.Module):
    """DCGAN Generator"""
    def __init__(self, latent_dim=100, channels=3, features_g=64):
        super(DCGANGenerator, self).__init__()
        self.latent_dim = latent_dim
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, features_g * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features_g * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g * 8, features_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g * 4, features_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g * 2, features_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.main(z)


class DCGANDiscriminator(nn.Module):
    """DCGAN Discriminator"""
    def __init__(self, channels=3, features_d=64):
        super(DCGANDiscriminator, self).__init__()
        
        self.main = nn.Sequential(
            nn.Conv2d(channels, features_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features_d, features_d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features_d * 2, features_d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features_d * 4, features_d * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features_d * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        return self.main(img).view(-1, 1).squeeze(1)


class WGANGenerator(nn.Module):
    """WGAN-GP Generator"""
    def __init__(self, latent_dim=100, channels=3, features_g=64):
        super(WGANGenerator, self).__init__()
        self.latent_dim = latent_dim
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, features_g * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features_g * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g * 8, features_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g * 4, features_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g * 2, features_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.main(z)


class WGANCritic(nn.Module):
    """WGAN-GP Critic"""
    def __init__(self, channels=3, features_d=64):
        super(WGANCritic, self).__init__()
        
        self.main = nn.Sequential(
            nn.Conv2d(channels, features_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features_d, features_d * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(features_d * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features_d * 2, features_d * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(features_d * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features_d * 4, features_d * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(features_d * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features_d * 8, 1, 4, 1, 0, bias=False)
        )
    
    def forward(self, img):
        return self.main(img).view(-1, 1).squeeze(1)


def weights_init(m):
    """Initialize network weights"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def gradient_penalty(critic, real, fake, device):
    """Calculate gradient penalty for WGAN-GP"""
    batch_size, c, h, w = real.shape
    epsilon = torch.rand((batch_size, 1, 1, 1)).repeat(1, c, h, w).to(device)
    interpolated_images = real * epsilon + fake * (1 - epsilon)
    interpolated_images.requires_grad_(True)
    
    mixed_scores = critic(interpolated_images)
    
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty


# ============================================================================
# Training Functions
# ============================================================================

def train_dcgan(generator, discriminator, dataloader, num_epochs, device, 
                lr=0.0002, output_dir='outputs'):
    """Train DCGAN"""
    os.makedirs(output_dir, exist_ok=True)
    
    generator.apply(weights_init).to(device)
    discriminator.apply(weights_init).to(device)
    
    opt_gen = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_disc = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    scaler = torch.cuda.amp.GradScaler()
    
    fixed_noise = torch.randn(64, generator.latent_dim, 1, 1).to(device)
    g_losses, d_losses = [], []
    
    print("\n" + "="*80)
    print("Starting DCGAN Training")
    print("="*80)
    
    for epoch in range(num_epochs):
        epoch_g_loss = 0
        epoch_d_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for real in pbar:
            real = real.to(device)
            batch_size = real.size(0)
            
            real_labels = torch.ones(batch_size).to(device)
            fake_labels = torch.zeros(batch_size).to(device)
            
            # Train Discriminator
            with torch.cuda.amp.autocast():
                disc_real = discriminator(real)
                loss_disc_real = criterion(disc_real, real_labels)
                
                noise = torch.randn(batch_size, generator.latent_dim, 1, 1).to(device)
                fake = generator(noise)
                disc_fake = discriminator(fake.detach())
                loss_disc_fake = criterion(disc_fake, fake_labels)
                
                loss_disc = (loss_disc_real + loss_disc_fake) / 2
            
            opt_disc.zero_grad()
            scaler.scale(loss_disc).backward()
            scaler.step(opt_disc)
            
            # Train Generator
            with torch.cuda.amp.autocast():
                output = discriminator(fake)
                loss_gen = criterion(output, real_labels)
            
            opt_gen.zero_grad()
            scaler.scale(loss_gen).backward()
            scaler.step(opt_gen)
            scaler.update()
            
            epoch_g_loss += loss_gen.item()
            epoch_d_loss += loss_disc.item()
            
            pbar.set_postfix({'D': f'{loss_disc.item():.4f}', 'G': f'{loss_gen.item():.4f}'})
        
        g_losses.append(epoch_g_loss / len(dataloader))
        d_losses.append(epoch_d_loss / len(dataloader))
        
        # Save samples
        if (epoch + 1) % 5 == 0:
            generator.eval()
            with torch.no_grad():
                fake_images = generator(fixed_noise)
                save_image(
                    (fake_images + 1) / 2,
                    f'{output_dir}/dcgan_epoch_{epoch+1}.png',
                    nrow=8
                )
            generator.train()
    
    return generator, discriminator, g_losses, d_losses


def train_wgan_gp(generator, critic, dataloader, num_epochs, device,
                  lr=0.0002, lambda_gp=10, critic_iterations=5, output_dir='outputs'):
    """Train WGAN-GP"""
    os.makedirs(output_dir, exist_ok=True)
    
    generator.apply(weights_init).to(device)
    critic.apply(weights_init).to(device)
    
    opt_gen = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_critic = optim.Adam(critic.parameters(), lr=lr, betas=(0.5, 0.999))
    scaler = torch.cuda.amp.GradScaler()
    
    fixed_noise = torch.randn(64, generator.latent_dim, 1, 1).to(device)
    g_losses, c_losses = [], []
    
    print("\n" + "="*80)
    print("Starting WGAN-GP Training")
    print("="*80)
    
    for epoch in range(num_epochs):
        epoch_g_loss = 0
        epoch_c_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for real in pbar:
            real = real.to(device)
            batch_size = real.size(0)
            
            # Train Critic
            for _ in range(critic_iterations):
                with torch.cuda.amp.autocast():
                    noise = torch.randn(batch_size, generator.latent_dim, 1, 1).to(device)
                    fake = generator(noise)
                    
                    critic_real = critic(real)
                    critic_fake = critic(fake.detach())
                    gp = gradient_penalty(critic, real, fake.detach(), device)
                    loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + lambda_gp * gp
                
                opt_critic.zero_grad()
                scaler.scale(loss_critic).backward()
                scaler.step(opt_critic)
            
            # Train Generator
            with torch.cuda.amp.autocast():
                noise = torch.randn(batch_size, generator.latent_dim, 1, 1).to(device)
                fake = generator(noise)
                gen_fake = critic(fake)
                loss_gen = -torch.mean(gen_fake)
            
            opt_gen.zero_grad()
            scaler.scale(loss_gen).backward()
            scaler.step(opt_gen)
            scaler.update()
            
            epoch_g_loss += loss_gen.item()
            epoch_c_loss += loss_critic.item()
            
            pbar.set_postfix({'C': f'{loss_critic.item():.4f}', 'G': f'{loss_gen.item():.4f}'})
        
        g_losses.append(epoch_g_loss / len(dataloader))
        c_losses.append(epoch_c_loss / len(dataloader))
        
        # Save samples
        if (epoch + 1) % 5 == 0:
            generator.eval()
            with torch.no_grad():
                fake_images = generator(fixed_noise)
                save_image(
                    (fake_images + 1) / 2,
                    f'{output_dir}/wgan_epoch_{epoch+1}.png',
                    nrow=8
                )
            generator.train()
    
    return generator, critic, g_losses, c_losses


def plot_losses(dcgan_g, dcgan_d, wgan_g, wgan_c, output_path='loss_curves.png'):
    """Plot training losses"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(dcgan_g, 'b-', linewidth=2)
    axes[0, 0].set_title('DCGAN Generator Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(dcgan_d, 'r-', linewidth=2)
    axes[0, 1].set_title('DCGAN Discriminator Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(wgan_g, 'b-', linewidth=2)
    axes[1, 0].set_title('WGAN-GP Generator Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(wgan_c, 'r-', linewidth=2)
    axes[1, 1].set_title('WGAN-GP Critic Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nLoss curves saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Train DCGAN and WGAN-GP')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples to use')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--skip_dcgan', action='store_true', help='Skip DCGAN training')
    parser.add_argument('--skip_wgan', action='store_true', help='Skip WGAN-GP training')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Data
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    dataset = ImageDataset(args.dataset, transform=transform, max_samples=args.max_samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                           num_workers=2, pin_memory=True)
    
    # Train DCGAN
    if not args.skip_dcgan:
        dcgan_gen = DCGANGenerator()
        dcgan_disc = DCGANDiscriminator()
        dcgan_gen, dcgan_disc, dcgan_g_losses, dcgan_d_losses = train_dcgan(
            dcgan_gen, dcgan_disc, dataloader, args.epochs, device, 
            lr=args.lr, output_dir=args.output_dir
        )
        
        torch.save({
            'generator_state_dict': dcgan_gen.state_dict(),
            'discriminator_state_dict': dcgan_disc.state_dict(),
            'g_losses': dcgan_g_losses,
            'd_losses': dcgan_d_losses,
        }, f'{args.output_dir}/dcgan_final.pth')
        print("DCGAN model saved!")
    
    # Train WGAN-GP
    if not args.skip_wgan:
        wgan_gen = WGANGenerator()
        wgan_critic = WGANCritic()
        wgan_gen, wgan_critic, wgan_g_losses, wgan_c_losses = train_wgan_gp(
            wgan_gen, wgan_critic, dataloader, args.epochs, device,
            lr=args.lr, output_dir=args.output_dir
        )
        
        torch.save({
            'generator_state_dict': wgan_gen.state_dict(),
            'critic_state_dict': wgan_critic.state_dict(),
            'g_losses': wgan_g_losses,
            'c_losses': wgan_c_losses,
        }, f'{args.output_dir}/wgan_gp_final.pth')
        print("WGAN-GP model saved!")
    
    # Plot losses
    if not args.skip_dcgan and not args.skip_wgan:
        plot_losses(dcgan_g_losses, dcgan_d_losses, wgan_g_losses, wgan_c_losses,
                   f'{args.output_dir}/loss_curves.png')
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
