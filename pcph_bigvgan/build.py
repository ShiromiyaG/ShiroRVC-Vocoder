from types import SimpleNamespace

from pcph_bigvgan.models.generator import PCPHBigVGANGenerator


#: Upsampler filters for the 44.1 kHz layout's five stages, ``[4, 4, 4, 4, 2]``.
#: RefineGAN2's four-stage widths extended on their own reasoning: the first
#: stage short, where a long kernel reaches furthest into the padded edge, the
#: later ones long; the last repeats the one before it. Rolloffs as
#: ``generator.UPSAMPLE_ROLLOFF``, with no image above Nyquist. The config's
#: ``filter_width``, ``rolloff`` and ``filter_beta`` override them, and another
#: stage count keeps the generator's own (the older ``[5, 4, 4, 4]`` exports).
FILTER_WIDTH = (12, 24, 32, 48, 48)
ROLLOFF = (0.84, 0.92, 0.94, 0.94, 0.94)
FILTER_BETA = (6.0, 6.0, 6.0, 9.0, 9.0)


def build_vocoder(config: dict) -> PCPHBigVGANGenerator:
    """PCPH-BigVGAN from f0 and the *normalised* log mel (``normalize_mel``),
    with no speaker input. The raw log mel sits around -5 with a spread of
    ~3, several times the scale the trunk's init expects."""
    model = config["vocoder"]["model"]
    data = config["data"]
    stages = len(model["upsample_rates"])
    filters = {
        key: model[key] if key in model else default
        for key, default in (("filter_width", FILTER_WIDTH), ("rolloff", ROLLOFF),
                             ("filter_beta", FILTER_BETA))
        if key in model or len(default) == stages
    }
    return PCPHBigVGANGenerator(
        sample_rate=data["sample_rate"],
        upsample_rates=tuple(model["upsample_rates"]),
        upsample_initial_channel=model["upsample_initial_channel"],
        resblock_kernel_sizes=model["resblock_kernel_sizes"],
        resblock_dilation_sizes=model["resblock_dilation_sizes"],
        resblock=model["resblock"],
        antialias=model["antialias"],
        num_mels=data["n_mels"],
        gin_channels=0,
        source_noise_std=model["source_noise_std"],
        source_harmonics=model["source_harmonics"],
        source_branch=model["source_branch"],
        source_type=model.get("source_type", "sine"),
        source_random_start_phase=model.get("source_random_start_phase", True),
        source_noise_eq=model.get("source_noise_eq"),
        source_max_frequency=model.get("source_max_frequency"),
        source_pulsed_noise=model.get("source_pulsed_noise", False),
        noise_branch_bands=model.get("noise_branch_bands", 0),
        noise_branch_split_source=model.get("noise_branch_split_source", False),
        output_gain=model["output_gain"],
        stage_channels=model.get("stage_channels"),
        prenet_blocks=model.get("prenet_blocks", 0),
        deep_source_stages=model.get("deep_source_stages", 0),
        **filters,
    )


def build_discriminator(config: dict, use_checkpointing: bool = False):
    """The PCPH-BigVGAN recipe's discriminator (v4 + UnivHD + SAN).

    ``d_mrd_mel_cond`` conditions its spectrogram branches on the normalised
    mel the vocoder is given; the discriminator then takes it as ``cond``."""
    from pcph_bigvgan.setup import get_d_model

    settings = dict(config["vocoder"]["discriminator"])
    data = config["data"]
    if settings.pop("d_mrd_mel_cond", False):
        settings["mrd_mel_cond"] = dict(
            sample_rate=data["sample_rate"],
            n_mels=data["n_mels"],
            fmin=data["mel_fmin"],
            fmax=data["mel_fmax"],
        )
    hparams = SimpleNamespace(
        model=SimpleNamespace(**settings),
        data=SimpleNamespace(sample_rate=data["sample_rate"]),
    )
    return get_d_model(hparams, use_checkpointing)
