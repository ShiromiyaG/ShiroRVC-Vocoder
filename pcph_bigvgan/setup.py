"""Assembling a run: what happens once, before the loop in ``train.py``."""

import os

import torch

from pcph_bigvgan.paths import LOGS_DIR
from pcph_bigvgan.terminal import info
from pcph_bigvgan.messages import DISCRIMINATOR_COMPILE_ENABLED, DISCRIMINATOR_COMPILE_NO_CUDA, DISCRIMINATOR_COMPILE_NOT_SUPPORTED


def loader_workers(requested: int) -> int:
    """``requested`` DataLoader workers, capped at the CPUs this process may
    use, the limit PyTorch warns past (Colab has 2)."""
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = os.cpu_count() or 1
    return max(0, min(int(requested), available))


def _inductor_cache_dir():
    """One Inductor cache for every run, under ``logs/``.

    Kept out of the per-model folder on purpose: the compiled artefacts are a
    function of the code and the GPU, not of the experiment, so every run
    should be warming the same cache.
    """
    cache_dir = os.path.join(LOGS_DIR, ".torchinductor")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", cache_dir)
    return cache_dir

#: The branch layout the vocoder trains against; a config's ``d_version``
#: overrides it.
DISCRIMINATOR_ID = "v4"


def get_d_model(config, use_checkpointing):
    discriminator_id = DISCRIMINATOR_ID
    # JSON has no tuples, so a configured schedule arrives as nested lists.  The
    # discriminators normalise them themselves; passing them through untouched
    # keeps this function free of the branch layout.
    def setting(name, default=None):
        """``None``/absent means "use the default"; ``[]`` means "none of these".

        This was ``value or default``, which collapsed the two: a config asking
        for *no* period branches got the full set back, and there was no way to
        turn a whole family off from JSON.
        """

        value = getattr(config.model, name, None)
        return default if value is None else value

    from pcph_bigvgan.models.discriminator import (
        DISCRIMINATOR_VERSIONS,
        MPD_MSD_Combined,
    )

    # The registry names the branch layout directly -- ``v2`` is Applio's (8
    # periods), ``v4`` is RefineGAN2's (5 periods + 3 multi-band
    # spectrogram branches) -- so this only has to check that the name exists.  Checked rather than passed through: it used to be
    # ``"v3" if id == "mpd_msd_v3" else "v2"``, which handed Applio's v2 to any
    # id it did not recognise, and a vocoder registered against the wrong name
    # would have trained against the wrong discriminator with nothing said.
    #
    # ``d_version`` overrides it.  A RefineGAN2 config without that key built
    # ``v3`` before the registry moved to ``v4``, and its D checkpoint now fails
    # ``assert_periods_match``: set ``"d_version": "v3"`` to resume it.
    if discriminator_id not in DISCRIMINATOR_VERSIONS:
        raise ValueError(
            f"Unknown discriminator {discriminator_id!r}; "
            f"known: {sorted(DISCRIMINATOR_VERSIONS)}."
        )
    version = str(getattr(config.model, "d_version", None) or discriminator_id)
    # ``d_use_*`` switches a whole family off; ``d_periods``/``d_resolutions``
    # replace its content when it's on (``None`` keeps the preset's).
    return MPD_MSD_Combined(
        config.model.use_spectral_norm,
        use_checkpointing=use_checkpointing,
        version=version,
        periods=[] if not setting("d_use_periods", True) else setting("d_periods"),
        resolutions=(
            [] if not setting("d_use_resolutions", True) else setting("d_resolutions")
        ),
        frequency_strides=setting("d_frequency_strides"),
        use_msd=bool(setting("d_use_msd", True)),
        # Opt-in, so an absent key builds the branches every existing
        # discriminator was trained with.
        use_fast_mpd=bool(setting("d_use_fast_mpd", False)),
        # On by default: the spectrogram branches' STFT and first conv run
        # outside FP16 autocast, where their unnormalised magnitude overflowed
        # at a raised learning rate.  ``false`` restores the all-FP16 path.
        mrd_fp32_input=bool(setting("d_mrd_fp32_input", True)),
        # ``None`` keeps what the version says; see ``MULTIBAND_MRD_VERSIONS``
        # and ``MPD_PRE_EMPHASIS_BY_VERSION``.
        mrd_multiband=setting("d_mrd_multiband", None),
        mrd_channels=int(setting("d_mrd_channels", 32)),
        mpd_pre_emphasis=setting("d_mpd_pre_emphasis", None),
        # ``None`` keeps the version's gamma; see ``R1_GAMMA_BY_VERSION``.
        r1_gamma=setting("d_r1_gamma", None),
        r1_interval=int(setting("d_r1_interval", 16)),
        r1_batch_fraction=float(setting("d_r1_batch_fraction", 0.5)),
        r1_segment=int(setting("d_r1_segment", 6400)),
        # UnivHD (arXiv 2512.03486) is opt-in and *additive*: it appends a
        # harmonic-order branch and removes nothing, which is how the paper
        # runs it.  Off by default because it is unmeasured on this fork -- the
        # gains it reports are on harmonic structure and F0RMSE for singing,
        # not on the frame-rate mirroring, which its ERB bandwidths are too
        # wide to separate above ~700 Hz.
        sample_rate=int(config.data.sample_rate),
        use_univhd=bool(setting("d_use_univhd", False)),
        # SAN (arXiv 2301.12811): splits every branch's last projection into a
        # unit-norm direction and a scale.  Off by default and unmeasured on
        # this fork -- it changes conv_post's state-dict keys, so it is a
        # fresh-run decision, and it changes the loss floor, so every number
        # read off loss_disc has to be relearned.
        use_san=bool(setting("d_use_san", False)),
        univhd_n_fft=int(setting("d_univhd_n_fft", 2048)),
        univhd_hop_length=int(setting("d_univhd_hop_length", 256)),
        univhd_harmonics=int(setting("d_univhd_harmonics", 10)),
        univhd_bins_per_octave=int(setting("d_univhd_bins_per_octave", 24)),
        univhd_f_min=float(setting("d_univhd_f_min", 80.0)),
        univhd_channels=int(setting("d_univhd_channels", 32)),
        univhd_half_harmonic=bool(setting("d_univhd_half_harmonic", True)),
        univhd_max_hz=(
            None
            if setting("d_univhd_max_hz", None) is None
            else float(setting("d_univhd_max_hz", None))
        ),
        # ``None`` keeps the version's pinned weight; a config key overrides it
        # on any version.  See ``UNIVHD_WEIGHT_BY_VERSION``.
        univhd_weight=(
            None
            if setting("d_univhd_weight", None) is None
            else float(setting("d_univhd_weight", None))
        ),
        msd_weight=float(setting("d_msd_weight", 1.0)),
        # Set by ``build_discriminator`` from the vocoder's mel.
        mrd_mel_cond=setting("mrd_mel_cond", None),
    )


def apply_precision_policy(net_g, amp_dtype):
    """Under autocast, keep the generator's precision-critical paths in FP32.

    BF16 needs it for mantissa, FP16 for range.  Sets ``fp32_residuals`` on
    every module that declares it.  Must run before the compile: Dynamo guards
    on the attribute.
    """
    enabled = amp_dtype in (torch.bfloat16, torch.float16)
    model = net_g.module if hasattr(net_g, "module") else net_g
    for module in model.modules():
        if hasattr(module, "fp32_residuals"):
            module.fp32_residuals = enabled


def normalize_san_weights(net_d, optimizer=None):
    """Put every SAN direction back on the unit sphere after ``optim_d.step``.

    The projection is only unit-norm because it is *kept* there; an Adam step
    moves it off, and a direction that is not a direction is the one thing the
    method cannot tolerate.  A no-op when no branch carries a SAN head.

    ``optimizer`` also projects a schedule-free ``z``: otherwise it drifts off
    the sphere and every step pulls the live weight after it.
    """

    model = net_d.module if hasattr(net_d, "module") else net_d
    for module in model.modules():
        normalize = getattr(module, "normalize_weight", None)
        if normalize is None:
            continue
        normalize()
        z = optimizer.state.get(module.weight, {}).get("z") if optimizer is not None else None
        if z is not None:
            with torch.no_grad():
                norm = z.flatten(1).norm(p=2, dim=1).clamp_min(1e-12)
                z.div_(norm.view((-1,) + (1,) * (z.ndim - 1)))


def enable_discriminator_compile(net_d, config, device, rank):
    """Compile the discriminator, driven by ``compile_discriminator`` in the
    config (not a run-spec flag, since it travels with the architecture).
    ``torch_compile_mode`` is not consulted: ``reduce-overhead`` records CUDA
    graphs, which this loop can't support, so the mode is fixed to plain
    fusion.
    """
    if not bool(getattr(config.train, "compile_discriminator", False)):
        return False
    if device.type != "cuda":
        if rank == 0:
            info(DISCRIMINATOR_COMPILE_NO_CUDA, tag="[INIT]")
        return False

    _inductor_cache_dir()

    model = net_d.module if hasattr(net_d, "module") else net_d
    enable = getattr(model, "enable_compile", None)
    if enable is None:
        if rank == 0:
            info(DISCRIMINATOR_COMPILE_NOT_SUPPORTED, tag="[INIT]")
        return False
    enabled = enable(mode="default")
    if enabled and rank == 0:
        info(DISCRIMINATOR_COMPILE_ENABLED.format(mode="default"), tag="[INIT]")
    return enabled
