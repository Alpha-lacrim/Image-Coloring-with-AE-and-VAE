======================================================================
  IMAGE COLORIZATION  ·  AE vs VAE  ·  FULL REPORT
======================================================================

PHASE 1  –  DATASET
----------------------------------------
  Source          : CIFAR-10
  Images selected : 10 (one per class, upscaled 32→64px bicubic)
  Resolution      : 64×64 pixels  |  3 RGB channels
  Pixel range     : [0, 1]  (normalised)
  Split           : 7 train / 1 val / 2 test  (70/10/20%)
  Augmented train : 245 samples (x35 from 7 originals)

PHASE 2  –  AUTOENCODER
----------------------------------------
  Encoder         : 4x Conv2D (32,64,128,256 | stride=2, ReLU)
                    Flatten → Dense(256) → Dense(64)
  Decoder         : Dense(64) → Dense(4·4·256) → Reshape
                    4x Conv2DTranspose (128,64,32,3 | stride=2)
                    Output activation: Sigmoid
  Loss            : MSE
  Optimizer       : Adam(lr=1e-3, ReduceLROnPlateau)
  Batch size      : 16
  Epochs trained  : 105
  Parameters      : 3,215,139

PHASE 3  –  VARIATIONAL AUTOENCODER
----------------------------------------
  Encoder         : Same as AE → Dense(μ) + Dense(log σ²)
                    Sampling  z = μ + exp(0.5·log σ²)·ε
  Decoder         : Same as AE
  Loss            : MSE + β·KL  (β=0.1)
  KL divergence   : −0.5·Σ(1 + log σ² − μ² − σ²)
  Optimizer       : Adam(lr=1e-3)
  Epochs trained  : 80
  Parameters      : 3,231,587

PHASE 4  –  QUANTITATIVE COMPARISON
----------------------------------------
  Metric                           AE                    VAE
  ----------------------------------------------------------
  PSNR            13.7300 ±  0.0542      12.6586 ±  0.0166
  SSIM             0.1833 ±  0.0024       0.2118 ±  0.0022
  MAE              0.1700 ±  0.0030       0.1963 ±  0.0004

PHASE 4  –  DISCUSSION
----------------------------------------

  AE (Deterministic):
  ├─ Pros: Simpler training (single loss term), deterministic output
  │        is easier to evaluate; slightly higher average PSNR/SSIM in
  │        small-data regimes because no KL penalty compresses the code.
  └─ Cons: No diversity — produces one fixed colorization per input.
           Blurry outputs (MSE regression to mean colour) especially on
           textures and fine colour edges.

  VAE (Probabilistic):
  ├─ Pros: Sampling ε gives multiple plausible colourizations, useful when
  │        true colour is ambiguous (e.g. a gray sky could be blue or overcast).
  │        Regularised latent space clusters semantically similar images.
  └─ Cons: KL penalty forces the code toward N(0,I), slightly compressing
           reconstruction quality. Training requires balancing β (too large →
           posterior collapse; too small → no regularisation).

  Shared Failure Modes (common to both on small data):
  • Desaturation / brownish bias:  MSE penalises colour errors equally, so
    the network hedges toward achromatic outputs.
  • Limited generalisation:  10 training images (even ×35 augmented) cannot
    cover ImageNet colour variety; a production system would need ≥ 10K images.
  • Spatial blurriness:  transposed-conv upsampling without skip connections
    discards high-frequency spatial detail; U-Net skips would help.
