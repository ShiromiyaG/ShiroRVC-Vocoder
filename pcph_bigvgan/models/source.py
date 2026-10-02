import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


#: Interpolation filter for the trunk's upsamplers, one entry per stage.
#:
#: Zero-stuffing by ``factor`` copies the input spectrum to every multiple of
#: the input rate; what the filter leaves of those copies is an *image* at
#: ``|k*R_in +- f|``.  That is imaging rather than aliasing, and only this
#: filter touches it -- but an image at ``k*R_in - j*f0`` moves against f0
#: exactly like a fold does, so the two are indistinguishable in a spectrogram.
#:
#: These were a flat ``12 / 0.90 / 6.0``, which left the worst image at
#: **-37.1 dB** while attenuating the partial that made it by 13.0 dB.
#: Measured on stage 3 with a partial at 0.95 of the input Nyquist,
#: image / passband:
#:
#:     w12 r0.90 b6.0   -37.1 / -13.0    97 taps
#:     w24 r0.95 b6.0   -68.4 /  -6.0   193
#:     w32 r0.97 b6.0   -68.0 /  -2.0   257
#:     w48 r0.99 b9.0   -94.4 /  -0.1   385
#:
#: The last stage's rolloff came down from 0.99 to 0.97 on 2026-09-10.  The row
#: above measures the image of a partial at 0.95 of the input Nyquist, which is
#: deep in the stopband; the image that matters is the one from content just
#: *under* the mirror, which lands just over it, inside the transition band
#: where the design has barely started attenuating.  Read off the kernel's own
#: response, as a fraction of the mirror -- the numbers are the same at any
#: factor, since the kernel is defined in normalised frequency, so at
#: ``[5,4,4,4]``'s last stage the mirror is 4000 Hz and these land at 4050 and
#: 4150:
#:
#:     rolloff   mirror+1.25%   +3.75%   |   -6.25%   -1.25%
#:     0.99         -17.8       -47.1    |   -0.01     -5.22
#:     0.97         -38.8      -127.0    |   -0.42    -14.44
#:     0.95         -90.7       -92.9    |   -2.70    -32.26
#:
#: 21 dB on the near image for 0.4 dB at 6% under the mirror is the trade 0.97
#: takes.  0.95 buys another 52 dB and starts costing real passband, so it is
#: where to go next if the near image is still what a measurement finds.  At
#: ``[5,4,4,4]`` this also lowers the ceiling the last block synthesises from,
#: 3960 -> 3880 Hz, which the overfit probe above says is not binding.
#:
#: Injecting a known sinusoid after each ``upsample_blocks[k]`` and subtracting
#: a clean render gives the path gain from that stage to the output directly:
#:
#:     stage 1 image -37.1 dB, path -59.3 dB  ->  -96.4 dB at the output
#:     stage 2 image -37.1     path -49.5     ->  -86.6
#:     stage 3 image -37.1     path -19.9     ->  -57.0
#:
#: which is why stage 3 was lengthened first.  Stages 1 and 2 follow anyway,
#: because that measurement is of the *linear* path and the images do not only
#: travel it: an image at ``2000 - f`` leaving stage 2 enters stage 3 and meets
#: dozens of ``leaky_relu``, and its second-order product with a strong partial
#: lands at ``2000 - (j+1)*f0`` with an amplitude set by harmonic energy rather
#: than by -49.5 dB of path gain.
#:
#: Stage 0 is left short.  What leaves it still has three stages of filtering
#: ahead of it, and it is the one stage where the input is short enough that a
#: long kernel costs edge: ``AntiAliasedUpsample1d`` pads, so a longer kernel
#: reaches further into an invented continuation.  Fraction of a 0.4 s training
#: segment whose samples that padding corrupts by more than 1%:
#:
#:     stage 0   40 in, 121 taps   21.0%    <- already the worst, left alone
#:     stage 1  200 in, 193 taps    9.2%
#:     stage 2  800 in, 257 taps    3.9%
#:     stage 3 3200 in, 385 taps    0.3%
DEFAULT_UPSAMPLE_WIDTH = (12, 24, 32, 48)
DEFAULT_UPSAMPLE_BETA = (6.0, 6.0, 6.0, 9.0)

#: Frames the excitation gain reads from ``z``.  More than one, so the prior's
#: frame-independent draw is averaged before it modulates the source.
SOURCE_GAIN_KERNEL = 5


class UnitNorm(nn.Module):
    """Weight norm without the gain: every output channel's filter at unit norm.

    This parametrizes ``conv_post``, and the reason is the gain it removes.
    Under ``weight_norm`` that layer's ``g`` is one scalar standing between the
    whole trunk and the ``tanh``, and every nonlinearity before it is a
    ``leaky_relu``, so the loss sees only the *product* of ``g`` and the
    trunk's scale.  Nothing in the objective decides the split; the optimizer
    decides it.  Each stage's ``input_conv`` is a plain conv, and under Adam a
    plain weight's norm grows by random walk, about ``lr * sqrt(fan_in *
    steps)`` per row -- predicted against measured row norms:

        run                          stage 0         stage 3
        this decoder, 2e-4, 98k     4.2 / 3.27      1.5 / 2.11
        RefineGAN,    1e-4, 53k     1.5 / 1.42      0.54 / 0.47

    Four stages in series multiply that, and ``g`` shrinks to keep the output
    at its level.  The first 32 kHz pretrain ended at ``g = 2.9e-4`` from
    0.574 at init.  ``dL/dg`` grows with the trunk's features, so that one
    scalar carried 99.9% of the generator's gradient norm (~14k, from the
    optimizer's second moment).  BF16 carried it; the first FP16 finetune of
    that pretrain went to NaN.

    With the norm fixed there is no such direction: the trunk's last features
    *are* the output's amplitude, so their scale is something the loss pins
    rather than something the optimizer drifts along.  Only the direction is
    learned.
    """

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return weight / torch.linalg.vector_norm(
            weight, dim=tuple(range(1, weight.dim())), keepdim=True
        )


#: Moving-average window the output's DC is taken from, in seconds.  At 0.1 s
#: the highpass it makes ripples by under 0.4 dB from 80 Hz up.
DC_WINDOW_SECONDS = 0.1


def remove_dc(x: torch.Tensor, window: int) -> torch.Tensor:
    """``x`` minus its centred moving average, shortened at the edges."""
    length = x.shape[-1]
    half = window // 2
    # Float64: a running sum over minutes of audio would lose the mean in FP32.
    total = F.pad(x.double().cumsum(-1), (1, 0))
    index = torch.arange(length, device=x.device)
    low = (index - half).clamp(min=0)
    high = (index + half + 1).clamp(max=length)
    mean = (total[..., high] - total[..., low]) / (high - low)
    return x - mean.to(x.dtype)


#: What a ``conv_post`` with a learned gain leaves in a state dict: the new
#: ``weight_norm`` API's gain, then the old one's.
LEGACY_OUTPUT_GAIN_KEYS = (
    "conv_post.parametrizations.weight.original0",
    "conv_post.weight_g",
)


def _refuse_learned_output_gain(
    module, state_dict, prefix, local_metadata, strict, missing_keys,
    unexpected_keys, error_msgs,
):
    """Load hook: refuse a decoder trained with a gain on ``conv_post``.

    Such a checkpoint cannot be converted.  Its trunk was trained against that
    gain, so without it the output is ``1 / g`` too loud -- ~3400x on the
    pretrain that motivated ``UnitNorm``.  And inference loads non-strictly, so
    without this the gain key would be dropped as unexpected, ``conv_post``
    left at its random init, and the old trunk's inflated features would drive
    the ``tanh`` into clipping without a single error.
    """

    found = [
        prefix + key
        for key in LEGACY_OUTPUT_GAIN_KEYS
        if prefix + key in state_dict
    ]
    if found:
        error_msgs.append(
            f"'{found[0]}' is a learned output gain, and this decoder's conv_post "
            "has none (unit-norm; see UnitNorm). This "
            "decoder was trained against that gain and cannot be converted -- "
            "train a fresh pretrain."
        )


def _match_output_gain(
    module, state_dict, prefix, local_metadata, strict, missing_keys,
    unexpected_keys, error_msgs,
):
    """Load hook for ``output_log_gain``.  Into a decoder built without one it
    is refused: a non-strict load would drop it and render ``exp(s)`` off in
    level.  Missing from a decoder built with one, it starts at 0 -- a gain of
    1, the output the checkpoint was trained to give."""

    key = prefix + "output_log_gain"
    if not getattr(module, "has_output_gain", False):
        if key in state_dict:
            error_msgs.append(
                f"'{key}' is a learned output gain but this decoder was built "
                "without one; set the config's output_gain option to load it."
            )
    elif key not in state_dict:
        state_dict[key] = torch.zeros(())


class SineGenerator(nn.Module):
    """Sine + additive-noise harmonic excitation source.

    On a fixed trunk over 3 seeds (held-out multi-scale mel, lower better):
    sine 1.9714 -> **1.7418** with ``source_gain`` on, bank 1.7357 -> 1.8057,
    comb 1.9649 -> 1.8363.

    ``comb`` and ``bank`` were removed on 2026-09-03: neither had beaten the
    sine once the excitation gain was on.  ``excitation_source`` in
    the checkpoint guards name the mismatch if a checkpoint from another
    source is loaded here.

    ``harmonic_num`` (2026-09-09)
    ----------------------------
    At 0 -- the default this shipped with, and Applio's -- the excitation is
    one partial, so *every* harmonic in the output is manufactured by the
    trunk.  This knob was added while chasing renders whose harmonics stopped
    around 6 kHz, on the theory that one partial under the trunk's 3960 Hz
    ceiling left nothing to build on above it.

    It ships at 0 because that theory did not survive measurement: overfitting
    a single clip, this decoder at ``harmonic_num=0`` reproduces a target's
    harmonic contrast to within 0.7 dB all the way to 13 kHz, under either
    stage layout.  See ``RefineGAN2Generator`` for the table.  A source short
    of partials is not what costs a *trained* model its high harmonics, so
    this is an escape hatch rather than a fix -- raising it puts scaffolding
    into the source, where it is alias-free and in tune by construction.

    ``harmonic_tilt`` gives partial ``j`` an amplitude of ``j ** -tilt``, so
    the source arrives with a slope instead of flat.  1.0 is a sawtooth's
    -6 dB/octave, and it lands close to a real voice (levels relative to the
    300-600 Hz band, f0=200):

        band          real voice    tilt 1.0
        1-2 kHz          -5.5         -9.1
        12-15.5 kHz     -33.8        -28.9

    Within ~5 dB across the whole band.  A single power law cannot match both ends -- a voice's slope steepens with
    frequency and this one does not -- which is what the knob is for; 1.17
    matches the top and costs 5 dB at 1-2 kHz.

    And a partial above Nyquist is *removed*, not folded: ``_f02sine`` takes
    ``f / sr`` mod 1, so ``j * f0`` past ``sr / 2`` aliases down onto the band
    as an inharmonic line that walks against f0.  The fade below zeroes it
    over the top ``NYQUIST_TAPER`` of the band rather than switching it off,
    because a hard gate clicks every time f0 drifts across the boundary.

    ``dim`` sizes ``merge.0.weight``, so this is a state-dict shape and a
    fresh pretrain -- and ``decoder_layout`` carries ``harmonic_tilt``, which
    is *not* in the state dict at any count.
    """

    #: Fraction of Nyquist over which a partial fades out instead of being
    #: switched off.  f0 moves between frames, so a hard mask makes the top
    #: partial blink on and off around the boundary -- a click at the frame
    #: rate on exactly the harmonics that are hardest to hear it in.
    NYQUIST_TAPER = 0.1

    def __init__(
        self,
        samp_rate,
        harmonic_num=0,
        sine_amp=0.1,
        noise_std=0.003,
        voiced_threshold=0,
        harmonic_tilt=1.0,
        harmonic_phase="random",
        phase_jitter=0.0,
        random_start_phase=False,
    ):
        """``harmonic_phase`` ``coherent`` starts every partial at phase 0, so
        they add up to one pulse per period, as a glottal cycle does; ``random``
        draws a new offset per partial on every call, and above ~4 kHz, where
        several partials share a critical band, that reads as roughness.

        ``phase_jitter`` draws each overtone's coherent offset uniformly within
        +- that many cycles on every call: an exactly repeating source let the
        discriminators separate within 150 steps.

        ``random_start_phase`` starts the fundamental anywhere in its cycle on
        every training call. Otherwise every crop begins at phase zero, and a
        phase-locked excitation puts its pulses at the same offset from the
        crop's start every time, which no real recording does."""
        super(SineGenerator, self).__init__()
        if harmonic_phase not in ("random", "coherent"):
            raise ValueError(
                f"harmonic_phase must be 'random' or 'coherent', not {harmonic_phase!r}."
            )
        self.harmonic_phase = harmonic_phase
        self.phase_jitter = float(phase_jitter)
        self.random_start_phase = bool(random_start_phase)
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.harmonic_tilt = float(harmonic_tilt)

        # Non-persistent: ``m_source`` owns exactly one state-dict key and
        # that is pinned by a test.  A buffer that changes the signal and
        # leaves no key is a ``decoder_layout`` field, like the upsamplers'
        # interpolation design.
        orders = torch.arange(1, self.dim + 1, dtype=torch.float32)
        self.register_buffer("harmonic_order", orders, persistent=False)
        self.register_buffer(
            "harmonic_gain",
            orders ** (-self.harmonic_tilt),
            persistent=False,
        )

        self.merge = nn.Sequential(
            nn.Linear(self.dim, 1, bias=False),
            nn.Tanh(),
        )
        # One scalar decides the excitation's amplitude, and it used to be
        # drawn from ``U(-1, 1)`` -- ``nn.Linear``'s default init at
        # ``fan_in = 1``.  With ``harmonic_num = 0`` this layer has nothing to
        # merge: it is that draw multiplying the source, and nothing downstream
        # normalises it.  Over 200 seeds it landed anywhere in +-0.99, negative
        # in 48% of them and under 0.1 in 14%, a 719x spread between the
        # loudest and the quietest; measured on the excitation itself, five
        # seeds gave RMS 0.0005 / 0.0364 / 0.0162 / 0.0700 / 0.0084 -- 132x,
        # with the low end a source that is effectively muted while the trunk
        # tries to learn from it.
        #
        # ``sine_amp`` is what states the intended amplitude, so 1.0 is what
        # lets it: RMS 0.0707 on every seed instead of one seed in seven
        # starting an order of magnitude down.  It also pins the ``Tanh``
        # above at -61.6 dB against its best linear fit, which is the whole
        # reason that ``Tanh`` needs no attention -- at 3x the gain it is
        # -42.9, and what it makes there is harmonic anyway.
        #
        # This is not the fresh-pretrain kind of change: ``merge.0.weight`` is
        # a state-dict key, so a resumed run loads what it learned and only new
        # runs start anywhere different.
        #
        # At ``dim > 1`` the harmonics sum rather than replace each other and
        # the ``Tanh`` is what bounds the total -- which is what it is for in
        # the source module this comes from.  Ones is still the right init
        # there: ``harmonic_gain`` has already set each partial's level, so
        # this layer only has to add them up, and the sum stays small enough
        # for the ``Tanh`` to leave alone.  At tilt 1.0 the partials are
        # phase-randomised, so the total is ``sine_amp * sqrt(sum 1/j^2)``,
        # which converges: measured 0.0707 RMS at 1 partial, 0.0890 at 16 and
        # 0.0898 at 32 -- 2.1 dB, and bounded above by 0.0907 at any count.
        # The ``Tanh``'s departure from its best linear fit follows: -61.6 dB
        # at 1, -47.1 at 16, -53.8 at 32 (it tracks the peak factor of the
        # phase draw, not the count).  What it makes is harmonic and lands on
        # partials the source already has.
        nn.init.ones_(self.merge[0].weight)

        # Octave band of each partial (1 | 2-3 | 4-7 | ...), so ``source_gain``
        # can shape the tilt with a handful of channels at any harmonic count.
        band = torch.floor(torch.log2(orders)).long()
        self.band_count = int(band.max()) + 1
        #: (dim, band_count) one-hot: sums the partials into their bands.
        self.register_buffer(
            "band_matrix", F.one_hot(band, self.band_count).float(), persistent=False
        )
        #: Noise gain channels: a first-order lowpass of the draw, the draw
        #: itself, and a first-order highpass of it, each with its own gain.
        #: One gain on white noise can only move the floor as a block, and what
        #: the floor is short of is *tilt* -- measured against a reference, too
        #: little above 8 kHz and too much below 1.5.  Mixing the three is a
        #: slope the projection can choose per frame.
        self.noise_gain_channels = 3
        #: Gain channels ``forward`` takes: one per octave band, then the noise.
        self.gain_channels = self.band_count + self.noise_gain_channels

    def _f02uv(self, f0):
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def _nyquist_fade(self, f0_buf):
        """Per-partial gain that reaches 0 at Nyquist. 1.0 everywhere at dim 1.

        ``f0_buf`` holds each partial's own frequency, so this is read off the
        partial rather than off the fundamental: which harmonics survive
        depends on the note, and at ``harmonic_num = 0`` nothing is ever near
        the boundary and this is exactly ``ones``.
        """

        nyquist = self.sampling_rate / 2.0
        return (
            (nyquist - f0_buf).div_(nyquist * self.NYQUIST_TAPER).clamp_(0.0, 1.0)
        )

    # Inductor cannot compile the cumsum over the sample axis: it lowers it to
    # a ``SplitScan`` whose codegen raises ``TypeError: list indices must be
    # integers or slices, not NoneType`` -- reproduced on torch 2.10 + cu130,
    # RTX 5060 -- and a failure inside the compiled region takes the *whole*
    # decoder down with it.  Only the phase stays out: it is one channel,
    # while everything after it is a chain over (batch, length, dim) that the
    # graph fuses.
    @torch.compiler.disable
    def _phase(self, f0):
        """The fundamental's phase in cycles, (batch, length, 1), and each
        partial's random initial phase, (batch, dim)."""
        # rad_values is F0 in rad mod 1 (the integer cycle count doesn't affect phase).
        # Scanned as (batch, length): over dim 1 of (batch, length, 1) CUDA
        # runs the cumsum as an outer-dim scan, parallel over that size-1 axis.
        rad_values = (f0[..., 0] / self.sampling_rate) % 1

        # random initial phase per harmonic, none for the fundamental
        if self.harmonic_phase == "coherent":
            rand_ini = torch.zeros(f0.shape[0], self.dim, device=f0.device)
            if self.phase_jitter > 0:
                rand_ini.uniform_(-self.phase_jitter, self.phase_jitter)
                rand_ini[:, 0] = 0
        else:
            rand_ini = torch.rand(f0.shape[0], self.dim, device=f0.device)
            rand_ini[:, 0] = 0

        tmp_over_one = torch.cumsum(rad_values, -1) % 1
        tmp_over_one_idx = (tmp_over_one[:, 1:] - tmp_over_one[:, :-1]) < 0
        cumsum_shift = torch.zeros_like(rad_values)
        cumsum_shift[:, 1:] = tmp_over_one_idx * -1.0
        phase = torch.cumsum(rad_values + cumsum_shift, dim=-1)[..., None]
        if self.random_start_phase and self.training:
            # Added before the partials multiply it, so it is a time shift.
            phase = phase + torch.rand(phase.shape[0], 1, 1, device=phase.device)
        return phase, rand_ini

    def _f02sine(self, f0):
        """f0: (batchsize, length, 1).  Returns (batchsize, length, dim) sines."""
        phase, rand_ini = self._phase(f0)

        # Partial j's phase is j times the fundamental's (mod 1), so the two
        # cumsums run on one channel instead of ``dim``.
        phase = torch.addcmul(rand_ini.unsqueeze(1), phase, self.harmonic_order)
        # In place from here: at ``dim`` partials every temporary is a full
        # (batch, length, dim) tensor.
        return phase.mul_(2 * np.pi).sin_()

    def forward(self, f0, gain=None):
        """f0: (batch, length, 1).  gain: (batch, length, gain_channels) or None."""
        with torch.no_grad():
            # Only the phase and the fade vary per sample *and* per partial.
            # Every other factor is applied after the sum over partials, where
            # it costs one channel instead of ``dim``.
            sine_waves = self._f02sine(f0)
            sine_waves.mul_(self._nyquist_fade(f0 * self.harmonic_order))

            uv = self._f02uv(f0)

            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = noise_amp * torch.randn_like(uv)

        # merge with grad
        if gain is not None:
            # Unit variance each, so the three start out interchangeable and a
            # gain reads as a level rather than as a filter's own scale.
            previous = F.pad(noise[:, :-1], (0, 0, 1, 0))
            lowpass = (noise + previous) * 0.7071067811865476
            highpass = (noise - previous) * 0.7071067811865476
            # No Tanh under a gain: a learned gain can push the sum into
            # saturation, and saturating a harmonic sum at the output rate
            # folds.  The gain used to multiply after the Tanh, so the output
            # is no less bounded than it was.
            noise = (
                lowpass * gain[..., -3:-2]
                + noise * gain[..., -2:-1]
                + highpass * gain[..., -1:]
            )

        if self.dim == 1:
            # The fade is exactly one here and ``harmonic_gain`` is ``[1.0]``,
            # so this is the single-partial body as it always was, bit for bit.
            sine_waves.mul_(self.sine_amp).mul_(uv)
            if gain is not None:
                sine_waves = sine_waves * gain[..., :1]
            merged = self.merge[0](sine_waves + noise)
        else:
            weight = self.merge[0].weight[0] * self.harmonic_gain * self.sine_amp
            if gain is None:
                sine = sine_waves @ weight[:, None]
            else:
                # The gain is constant within an octave band, so each band is
                # summed first: no (batch, length, dim) gain tensor, and no
                # scatter-add of every partial into its band in the backward.
                bands = sine_waves @ (self.band_matrix * weight[:, None])
                sine = (bands * gain[..., : self.band_count]).sum(-1, keepdim=True)
            # ``merge`` over ``dim`` iid draws of std ``s / sqrt(dim)`` is one
            # draw of std ``s * ||w|| / sqrt(dim)``, so a single channel stands
            # in for all of them and the merged level stays ``noise_std`` at
            # init (``w`` is ones) whatever the harmonic count.
            noise_scale = self.merge[0].weight.norm() / self.dim**0.5
            merged = sine * uv + noise * noise_scale

        return merged if gain is not None else self.merge[1](merged)
