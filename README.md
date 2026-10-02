# PCPH-BigVGAN

Trainer for the PCPH-BigVGAN mel vocoder: BigVGAN v2's anti-aliased SnakeBeta
blocks at voice scale, driven by a pseudo-constant-power harmonic excitation
(every harmonic to Nyquist, phase-locked) built from f0. It renders a
normalised log mel plus f0 to 44.1 kHz audio.

The code is the vocoder trainer of ShiroRVC's rectified pipeline, extracted
with what it depends on and nothing else.

## Install

```
pip install -r requirements.txt
```

Triton is optional (Linux, CUDA): with it the SnakeBeta activations and the
upsamplers run as fused kernels.

## Data

A filelist with one clip per row:

```
path/to/clip.wav|path/to/clip.f0.npy
```

- audio at the config's sample rate (44.1 kHz), any format `soundfile` reads;
- f0 in Hz as a `.npy` at 100 frames per second, 0 where unvoiced.

An RVC `filelist.txt` (`audio|features|f0|f0_voiced|speaker`) works as it is:
the fourth field is read as the pitch. Relative paths start at `--data-root`.

## Train

```
python train.py --name my_vocoder --filelist data/filelist.txt --epochs 100 --batch-size 8 --precision bf16
```

Everything goes to `logs/my_vocoder/`:

- `config.json`, copied from `configs/44100.json` on the first run; edit it
  there to change the model or the losses of that run;
- `G_<step>.pth` and `D_<step>.pth`, the checkpoints a rerun resumes from;
- `my_vocoder_vocoder_<epoch>e_<step>s.pth`, the export (EMA weights and
  config, `kind: rectified_vocoder`), which ShiroRVC and the Applio port load;
- `eval/`, TensorBoard scalars, and previews of one clip rendered from its
  real mel: `tensorboard --logdir logs/my_vocoder/eval`.

`python train.py --help` lists the remaining options (several GPUs, starting
from a pretrain, degraded-mel training).
