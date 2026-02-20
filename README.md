# Deep Learning from Scratch — Notebooks

A collection of Jupyter notebooks implementing foundational deep learning architectures from scratch, covering **Autoencoders**, **Variational Autoencoders**, **Attention Mechanisms**, and **Generative Adversarial Networks**.

## Notebooks

### 1. Autoencoders (`Autoencoders.ipynb`)

Four autoencoder variants built with Keras on the MNIST dataset.

| Architecture | Description |
|---|---|
| Vanilla Autoencoder | Single hidden layer (784 → 64 → 784) |
| Multilayer Autoencoder | Deeper encoder-decoder (784 → 128 → 64 → 128 → 784) |
| Convolutional Autoencoder | Conv2D + MaxPooling encoder with UpSampling decoder |
| Regularized Autoencoder | Sparse (L1 activity regularization) and Denoising (Gaussian noise) variants |

### 2. Variational Autoencoders (`Variational_Autoencoders.ipynb`)

Extends the autoencoder framework with probabilistic latent spaces.

- **Simple Autoencoder** with 2D bottleneck for latent space visualization
- **Variational Autoencoder (VAE)** with reparameterization trick and KL divergence loss
- 2D latent space scatter plots colored by digit class
- Digit manifold generation by sampling from the learned latent distribution

### 3. Attention Mechanisms (`Attention_Mechanisms.ipynb`)

Custom Keras layers implementing a family of attention mechanisms, with three NLP tasks.

**Attention Layers:**

| Layer | Reference |
|---|---|
| Self-Attention (multi-hop, with optional penalization) | Lin et al., 2017 |
| Global (Soft) Attention | Bahdanau et al., 2015 |
| Local Attention (monotonic and predictive alignment) | Luong et al., 2015 |

**Score Functions:** Dot Product, Scaled Dot Product, General, Concat, Location-based

**Example Tasks:**

- Sentiment Classification — IMDB Reviews (binary, many-to-one)
- Document Classification — Reuters (multi-class, many-to-one)
- Text Generation — Shakespeare (character-level, many-to-one)
- Score Function Comparison and Attention Weight Visualization

### 4. PyTorch GANs (`PyTorch_GANs.ipynb`)

Eight GAN variants implemented in PyTorch, all trained on MNIST for easy comparison.

| GAN Variant | Key Innovation | Reference |
|---|---|---|
| Vanilla GAN | Original adversarial framework | Goodfellow et al., 2014 |
| DCGAN | Convolutional architecture guidelines | Radford et al., 2016 |
| Conditional GAN | Class-conditioned generation | Mirza & Osindero, 2014 |
| WGAN | Wasserstein distance + weight clipping | Arjovsky et al., 2017 |
| WGAN-GP | Gradient penalty for Lipschitz constraint | Gulrajani et al., 2017 |
| LSGAN | Least squares loss | Mao et al., 2017 |
| InfoGAN | Disentangled latent codes via mutual information | Chen et al., 2016 |
| Adversarial Autoencoder | GAN-regularized latent space | Makhzani et al., 2016 |

## Getting Started

### Prerequisites

```
Python 3.9+
```

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/attention-and-gan-notebooks.git
cd attention-and-gan-notebooks
pip install -r requirements.txt
```

### Requirements

```
tensorflow>=2.10
torch>=2.0
torchvision
numpy
matplotlib
scikit-learn
scipy
```

Optionally, for model visualization in the autoencoder notebooks:

```
pydot
graphviz
```

### Running

```bash
jupyter notebook
```

Open any `.ipynb` file and run cells sequentially.

## Project Structure

```
├── Autoencoders.ipynb              # Vanilla, Multilayer, Conv, Regularized AEs
├── Variational_Autoencoders.ipynb  # VAE with latent space visualization
├── Attention_Mechanisms.ipynb      # Self/Global/Local Attention + NLP tasks
├── PyTorch_GANs.ipynb              # 8 GAN variants on MNIST
├── requirements.txt
└── README.md
```

## Acknowledgments

- [Autoencoders notebook](https://towardsdatascience.com/deep-inside-autoencoders-7e41f319999f) — original blog post companion
- [uzaymacar/attention-mechanisms](https://github.com/uzaymacar/attention-mechanisms) — attention layer reference implementations
- [eriklindernoren/PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN) — PyTorch GAN collection

## License

MIT
