import gradio as gr
import torch
import torch.nn as nn
from torchvision.utils import make_grid
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io

# Model Architectures (copied from notebook)
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


# Global variables for models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dcgan_model = None
wgan_model = None


def load_models(dcgan_path='dcgan_final.pth', wgan_path='wgan_gp_final.pth'):
    """Load trained models"""
    global dcgan_model, wgan_model
    
    # Initialize models
    dcgan_model = DCGANGenerator(latent_dim=100).to(device)
    wgan_model = WGANGenerator(latent_dim=100).to(device)
    
    try:
        # Load DCGAN
        checkpoint = torch.load(dcgan_path, map_location=device)
        dcgan_model.load_state_dict(checkpoint['generator_state_dict'])
        dcgan_model.eval()
        
        # Load WGAN-GP
        checkpoint = torch.load(wgan_path, map_location=device)
        wgan_model.load_state_dict(checkpoint['generator_state_dict'])
        wgan_model.eval()
        
        return True, "Models loaded successfully!"
    except Exception as e:
        return False, f"Error loading models: {str(e)}"


def generate_images(model, num_images=16, seed=None):
    """Generate images from a model"""
    if model is None:
        return None
    
    if seed is not None:
        torch.manual_seed(seed)
    
    model.eval()
    with torch.no_grad():
        noise = torch.randn(num_images, model.latent_dim, 1, 1).to(device)
        fake_images = model(noise)
        fake_images = (fake_images + 1) / 2  # Denormalize to [0, 1]
        
        # Create grid
        grid = make_grid(fake_images, nrow=4, padding=2)
        
        # Convert to numpy and PIL
        grid_np = grid.cpu().numpy().transpose(1, 2, 0)
        grid_np = (grid_np * 255).astype(np.uint8)
        
        return Image.fromarray(grid_np)


def generate_dcgan(num_images, seed):
    """Generate images using DCGAN"""
    if dcgan_model is None:
        return None
    return generate_images(dcgan_model, num_images, seed)


def generate_wgan(num_images, seed):
    """Generate images using WGAN-GP"""
    if wgan_model is None:
        return None
    return generate_images(wgan_model, num_images, seed)


def generate_comparison(num_images, seed):
    """Generate side-by-side comparison"""
    if dcgan_model is None or wgan_model is None:
        return None
    
    dcgan_img = generate_images(dcgan_model, num_images, seed)
    wgan_img = generate_images(wgan_model, num_images, seed)
    
    # Create side by side comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    axes[0].imshow(dcgan_img)
    axes[0].set_title('DCGAN Generated Images', fontsize=16, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(wgan_img)
    axes[1].set_title('WGAN-GP Generated Images', fontsize=16, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # Convert to PIL
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    comparison_img = Image.open(buf)
    plt.close()
    
    return comparison_img


def interpolate_latent(model, num_steps=10, seed=None):
    """Generate interpolation between two latent vectors"""
    if model is None:
        return None
    
    if seed is not None:
        torch.manual_seed(seed)
    
    model.eval()
    with torch.no_grad():
        # Generate two random latent vectors
        z1 = torch.randn(1, model.latent_dim, 1, 1).to(device)
        z2 = torch.randn(1, model.latent_dim, 1, 1).to(device)
        
        # Interpolate
        alphas = torch.linspace(0, 1, num_steps).to(device)
        interpolated = []
        
        for alpha in alphas:
            z = (1 - alpha) * z1 + alpha * z2
            fake_image = model(z)
            fake_image = (fake_image + 1) / 2  # Denormalize
            interpolated.append(fake_image)
        
        # Create grid
        interpolated_tensor = torch.cat(interpolated, dim=0)
        grid = make_grid(interpolated_tensor, nrow=num_steps, padding=2)
        
        # Convert to PIL
        grid_np = grid.cpu().numpy().transpose(1, 2, 0)
        grid_np = (grid_np * 255).astype(np.uint8)
        
        return Image.fromarray(grid_np)


def plot_training_curves(dcgan_path='dcgan_final.pth', wgan_path='wgan_gp_final.pth'):
    """Plot training loss curves"""
    try:
        # Load losses
        dcgan_checkpoint = torch.load(dcgan_path, map_location='cpu')
        wgan_checkpoint = torch.load(wgan_path, map_location='cpu')
        
        dcgan_g_losses = dcgan_checkpoint['g_losses']
        dcgan_d_losses = dcgan_checkpoint['d_losses']
        wgan_g_losses = wgan_checkpoint['g_losses']
        wgan_c_losses = wgan_checkpoint['c_losses']
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # DCGAN
        axes[0, 0].plot(dcgan_g_losses, label='Generator', color='blue', linewidth=2)
        axes[0, 0].set_title('DCGAN Generator Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(dcgan_d_losses, label='Discriminator', color='red', linewidth=2)
        axes[0, 1].set_title('DCGAN Discriminator Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # WGAN-GP
        axes[1, 0].plot(wgan_g_losses, label='Generator', color='blue', linewidth=2)
        axes[1, 0].set_title('WGAN-GP Generator Loss', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(wgan_c_losses, label='Critic', color='red', linewidth=2)
        axes[1, 1].set_title('WGAN-GP Critic Loss', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to PIL
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plot_img = Image.open(buf)
        plt.close()
        
        return plot_img
    except Exception as e:
        return None


# Create Gradio Interface
with gr.Blocks(title="GAN Mode Collapse Demo: DCGAN vs WGAN-GP") as demo:
    gr.Markdown("""
    # 🎨 Tackling Mode Collapse in GANs
    ### Comparing DCGAN and WGAN-GP
    
    This demo showcases the differences between DCGAN and WGAN-GP in addressing mode collapse.
    **WGAN-GP** uses Wasserstein loss with gradient penalty to improve training stability and diversity.
    """)
    
    # Model Loading Section
    with gr.Row():
        load_btn = gr.Button("🔄 Load Models", variant="primary")
        load_status = gr.Textbox(label="Status", value="Models not loaded. Click 'Load Models' button.")
    
    load_btn.click(
        fn=lambda: load_models(),
        outputs=load_status
    )
    
    gr.Markdown("---")
    
    # Generation Tabs
    with gr.Tabs():
        # Tab 1: DCGAN Generation
        with gr.Tab("🔵 DCGAN"):
            gr.Markdown("### Generate images using DCGAN")
            with gr.Row():
                dcgan_num = gr.Slider(4, 64, value=16, step=4, label="Number of Images")
                dcgan_seed = gr.Number(value=42, label="Random Seed")
            dcgan_btn = gr.Button("Generate DCGAN Images", variant="primary")
            dcgan_output = gr.Image(label="Generated Images")
            
            dcgan_btn.click(
                fn=generate_dcgan,
                inputs=[dcgan_num, dcgan_seed],
                outputs=dcgan_output
            )
        
        # Tab 2: WGAN-GP Generation
        with gr.Tab("🟢 WGAN-GP"):
            gr.Markdown("### Generate images using WGAN-GP")
            with gr.Row():
                wgan_num = gr.Slider(4, 64, value=16, step=4, label="Number of Images")
                wgan_seed = gr.Number(value=42, label="Random Seed")
            wgan_btn = gr.Button("Generate WGAN-GP Images", variant="primary")
            wgan_output = gr.Image(label="Generated Images")
            
            wgan_btn.click(
                fn=generate_wgan,
                inputs=[wgan_num, wgan_seed],
                outputs=wgan_output
            )
        
        # Tab 3: Side-by-Side Comparison
        with gr.Tab("⚖️ Comparison"):
            gr.Markdown("### Side-by-side comparison of both models")
            with gr.Row():
                comp_num = gr.Slider(4, 64, value=16, step=4, label="Number of Images")
                comp_seed = gr.Number(value=42, label="Random Seed")
            comp_btn = gr.Button("Generate Comparison", variant="primary")
            comp_output = gr.Image(label="Model Comparison")
            
            comp_btn.click(
                fn=generate_comparison,
                inputs=[comp_num, comp_seed],
                outputs=comp_output
            )
        
        # Tab 4: Latent Space Interpolation
        with gr.Tab("🔄 Interpolation"):
            gr.Markdown("### Explore latent space through interpolation")
            with gr.Row():
                with gr.Column():
                    interp_model = gr.Radio(["DCGAN", "WGAN-GP"], value="WGAN-GP", label="Model")
                    interp_steps = gr.Slider(5, 20, value=10, step=1, label="Interpolation Steps")
                    interp_seed = gr.Number(value=42, label="Random Seed")
                    interp_btn = gr.Button("Generate Interpolation", variant="primary")
                with gr.Column():
                    interp_output = gr.Image(label="Latent Space Interpolation")
            
            def interpolate_wrapper(model_name, steps, seed):
                model = dcgan_model if model_name == "DCGAN" else wgan_model
                return interpolate_latent(model, steps, seed)
            
            interp_btn.click(
                fn=interpolate_wrapper,
                inputs=[interp_model, interp_steps, interp_seed],
                outputs=interp_output
            )
        
        # Tab 5: Training Curves
        with gr.Tab("📊 Training Analysis"):
            gr.Markdown("### Training loss curves comparison")
            curves_btn = gr.Button("Show Training Curves", variant="primary")
            curves_output = gr.Image(label="Loss Curves")
            
            curves_btn.click(
                fn=plot_training_curves,
                outputs=curves_output
            )
    
    gr.Markdown("""
    ---
    ### 🔬 Key Differences:
    
    | Feature | DCGAN | WGAN-GP |
    |---------|-------|---------|
    | **Loss Function** | Binary Cross Entropy | Wasserstein Distance |
    | **Discriminator Output** | Sigmoid (probability) | Linear (score) |
    | **Training Stability** | Can be unstable | More stable |
    | **Mode Collapse** | Susceptible | Reduced risk |
    | **Gradient Flow** | Can vanish | Improved with GP |
    | **Hyperparameter Sensitivity** | High | Lower |
    
    ### 📈 Expected Results:
    - **WGAN-GP** should show more diverse and stable image generation
    - **DCGAN** may exhibit mode collapse with repeated patterns
    - Training curves show WGAN-GP's superior stability
    
    ### 💡 Tips:
    - Use the same seed for fair comparison between models
    - Try different seeds to explore variety in generation
    - Interpolation reveals smoothness of learned latent space
    """)


if __name__ == "__main__":
    # Try to load models on startup
    success, message = load_models()
    print(message)
    
    # Launch app
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860
    )
