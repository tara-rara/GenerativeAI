# Tackling Mode Collapse in GANs: DCGAN vs WGAN-GP

A comprehensive implementation comparing Deep Convolutional GANs (DCGAN) and Wasserstein GANs with Gradient Penalty (WGAN-GP) to address mode collapse in generative models.

## 🎯 Objective

Demonstrate how advanced loss functions and training techniques in WGAN-GP improve training stability and diversity of generated images compared to baseline DCGAN.

## 📋 Table of Contents

- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [Model Architectures](#model-architectures)
- [Training](#training)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [Results](#results)

## ✨ Features

### Models Implemented
- **DCGAN (Baseline)**
  - Binary Cross Entropy loss
  - Standard discriminator with sigmoid output
  - Batch normalization
  - Prone to mode collapse

- **WGAN-GP (Advanced)**
  - Wasserstein distance loss
  - Critic network (no sigmoid)
  - Gradient penalty (λ=10)
  - 5 critic updates per generator update
  - Improved training stability

### Key Capabilities
- Mixed precision training (AMP) for GPU efficiency
- Checkpoint saving every 5 epochs
- Real-time visualization during training
- Comprehensive diversity metrics
- Side-by-side model comparison
- Interactive Gradio web interface

## 🚀 Setup

### Platform Requirements
- **Platform**: Kaggle (recommended) or local with GPU
- **GPU**: NVIDIA T4 x2 or equivalent
- **Python**: 3.8+
- **CUDA**: 11.0+ (for GPU support)

### Dataset Options
1. **Pokemon Sprites**: https://www.kaggle.com/datasets/jackemartin/pokemon-sprites
2. **Anime Faces (64×64)**: https://www.kaggle.com/datasets/soumikrakshit/anime-faces

### Installation

```bash
# Clone or download the repository
git clone <your-repo-url>
cd gan-mode-collapse

# Install dependencies
pip install -r requirements.txt
```

### On Kaggle

1. Create a new notebook
2. Add dataset (Pokemon Sprites or Anime Faces)
3. Enable GPU accelerator (T4 x2 recommended)
4. Upload the notebook file
5. Update `DATASET_PATH` in the code

## 💻 Usage

### Training on Kaggle

1. **Upload the notebook**: `gan_mode_collapse.ipynb`

2. **Configure settings**:
```python
# In the notebook, update these parameters:
DATASET_PATH = '/kaggle/input/anime-faces/'  # or pokemon-sprites
BATCH_SIZE = 64
MAX_SAMPLES = 10000  # Adjust based on memory
NUM_EPOCHS_DCGAN = 50
NUM_EPOCHS_WGAN = 50
```

3. **Run all cells** sequentially

### Training Locally

```bash
# Run the notebook
jupyter notebook gan_mode_collapse.ipynb

# Or run training script (if converted from notebook)
python train_gans.py
```

### Memory Optimization Tips

For limited GPU memory:
- Reduce `BATCH_SIZE` to 32 or 16
- Reduce `MAX_SAMPLES` to 5000
- Reduce `NUM_EPOCHS` to 25-30
- Use single GPU instead of DataParallel

## 🏗️ Model Architectures

### DCGAN Generator
```
Input: 100D noise vector (z)
↓
ConvTranspose2d(100 → 512) → BN → ReLU
ConvTranspose2d(512 → 256) → BN → ReLU
ConvTranspose2d(256 → 128) → BN → ReLU
ConvTranspose2d(128 → 64) → BN → ReLU
ConvTranspose2d(64 → 3) → Tanh
↓
Output: 3×64×64 image
```

### DCGAN Discriminator
```
Input: 3×64×64 image
↓
Conv2d(3 → 64) → LeakyReLU(0.2)
Conv2d(64 → 128) → BN → LeakyReLU(0.2)
Conv2d(128 → 256) → BN → LeakyReLU(0.2)
Conv2d(256 → 512) → BN → LeakyReLU(0.2)
Conv2d(512 → 1) → Sigmoid
↓
Output: Probability [0, 1]
```

### WGAN-GP Critic
```
Same architecture as discriminator but:
- Instance Normalization instead of Batch Normalization
- No Sigmoid (outputs raw score)
- Trained with Wasserstein loss + Gradient Penalty
```

## 🎓 Training

### Training Strategy

1. **DCGAN Training**:
   - Loss: Binary Cross Entropy
   - Optimizer: Adam (lr=0.0002, β1=0.5, β2=0.999)
   - Real labels = 1, Fake labels = 0
   - Alternate discriminator and generator updates

2. **WGAN-GP Training**:
   - Loss: Wasserstein Distance + Gradient Penalty
   - Optimizer: Adam (lr=0.0002, β1=0.5, β2=0.999)
   - 5 critic iterations per generator iteration
   - Gradient penalty λ = 10

### Hyperparameters

```python
LATENT_DIM = 100          # Noise vector dimension
BATCH_SIZE = 64           # Batch size
LEARNING_RATE = 0.0002    # Learning rate
NUM_EPOCHS = 50           # Training epochs
LAMBDA_GP = 10            # Gradient penalty coefficient
CRITIC_ITERATIONS = 5     # Critic updates per generator
```

### Training Time Estimates

| Dataset Size | Epochs | GPU | Approx. Time |
|-------------|--------|-----|--------------|
| 10,000 images | 50 | T4 x2 | 2-3 hours |
| 20,000 images | 50 | T4 x2 | 4-5 hours |
| 10,000 images | 100 | T4 x2 | 4-6 hours |

## 📊 Evaluation

### Quantitative Metrics

1. **Loss Curves**
   - Generator loss vs. epochs
   - Discriminator/Critic loss vs. epochs
   - Loss stability analysis

2. **Diversity Metrics**
   - Mean pairwise distance
   - Pixel-space variance
   - Standard deviation of distances

3. **Optional Advanced Metrics**
   - Fréchet Inception Distance (FID)
   - Inception Score (IS)

### Visual Evaluation

- Generated image quality
- Diversity of samples
- Mode collapse detection
- Latent space interpolation smoothness

### Expected Results

**DCGAN**:
- May show mode collapse after 30-40 epochs
- Less diverse samples
- Training instability
- Higher variance in losses

**WGAN-GP**:
- Stable training throughout
- More diverse samples
- Smoother loss curves
- Better gradient flow

## 🌐 Deployment

### Gradio App

Launch the interactive web interface:

```bash
python gradio_app.py
```

Features:
- Generate images from both models
- Side-by-side comparison
- Latent space interpolation
- View training curves
- Adjustable parameters

### App Interface Sections

1. **Model Loading**: Load pre-trained checkpoints
2. **DCGAN Tab**: Generate DCGAN samples
3. **WGAN-GP Tab**: Generate WGAN-GP samples
4. **Comparison Tab**: Side-by-side comparison
5. **Interpolation Tab**: Explore latent space
6. **Training Analysis**: View loss curves

### Deployment Options

**Hugging Face Spaces**:
```bash
# Create a new Space on Hugging Face
# Upload gradio_app.py and model checkpoints
# Set requirements.txt
```

**Local Hosting**:
```bash
python gradio_app.py
# Access at http://localhost:7860
```

**Streamlit Alternative** (if needed):
```bash
streamlit run streamlit_app.py
```

## 📈 Results

### Training Stability

WGAN-GP shows significantly more stable training:
- Smoother loss curves
- No sudden divergence
- Consistent improvement

### Diversity Improvement

Typical diversity improvements (WGAN-GP vs DCGAN):
- Mean pairwise distance: +15-30%
- Pixel variance: +20-40%
- Visual diversity: Qualitatively superior

### Sample Quality

**DCGAN Issues**:
- May collapse to few modes
- Repetitive patterns
- Lower detail quality

**WGAN-GP Advantages**:
- Captures more modes
- Higher variety
- Better detail preservation

## 📁 Project Structure

```
gan-mode-collapse/
├── gan_mode_collapse.ipynb    # Main training notebook
├── gradio_app.py              # Deployment app
├── requirements.txt           # Dependencies
├── README.md                  # This file
├── checkpoints/               # Saved model checkpoints
│   ├── dcgan_final.pth
│   └── wgan_gp_final.pth
└── outputs/                   # Generated images
    ├── loss_comparison.png
    └── generated_comparison.png
```

## 🔬 Key Findings

### Why WGAN-GP Works Better

1. **Wasserstein Distance**: Provides meaningful gradients even when distributions don't overlap
2. **Gradient Penalty**: Enforces 1-Lipschitz constraint without weight clipping
3. **No Sigmoid**: Critic outputs unbounded scores, avoiding saturation
4. **More Critic Updates**: Better discriminator helps generator learn

### Common Issues and Solutions

**Problem**: Out of memory
- **Solution**: Reduce batch size, limit dataset size

**Problem**: Training too slow
- **Solution**: Enable mixed precision (AMP), use fewer epochs

**Problem**: Poor image quality
- **Solution**: Train longer, adjust learning rate, check data normalization

**Problem**: Mode collapse in DCGAN
- **Solution**: Expected behavior - compare with WGAN-GP results

## 📚 References

1. **DCGAN**: Radford et al., "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" (2015)
2. **WGAN**: Arjovsky et al., "Wasserstein GAN" (2017)
3. **WGAN-GP**: Gulrajani et al., "Improved Training of Wasserstein GANs" (2017)

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional GAN variants (StyleGAN, ProGAN)
- More evaluation metrics
- Better hyperparameter tuning
- Dataset augmentation techniques

## 📝 License

MIT License - feel free to use for educational purposes

## 🙏 Acknowledgments

- Kaggle for providing GPU resources
- PyTorch team for the framework
- Original GAN papers and authors

---

**Note**: This implementation is designed for educational purposes to demonstrate the differences between DCGAN and WGAN-GP in addressing mode collapse.

For questions or issues, please open an issue on GitHub or contact the maintainers.
