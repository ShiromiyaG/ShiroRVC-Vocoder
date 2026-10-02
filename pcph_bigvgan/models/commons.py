import torch


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def expand_f0(f0: torch.Tensor, length: int) -> torch.Tensor:
    """f0 at the frame rate -> f0 at the output rate, (batch, 1, length).

    Interpolated in log Hz, so the frame-rate ripple is constant in cents
    rather than in Hz, and gated by a ``nearest`` voiced mask, so the boundary
    stays where the f0 estimator put it.  See ``RefineGAN2Generator._expand_f0``
    for what each of those fixed.

    The interpolation is *normalised* by the voiced mask.  Interpolating
    ``log f0`` directly blends a voiced frame with the ``log 1 = 0`` of an
    unvoiced neighbour, so the last half-frame before every voiced/unvoiced
    boundary glides geometrically toward 1 Hz while the gate still reads
    voiced.  Dividing by the interpolated mask weights the voiced side only,
    which holds the edge frame's pitch; between two voiced frames it is plain
    linear interpolation.
    """

    voiced = (f0 > 0).to(f0.dtype)
    log_f0 = torch.log(f0.clamp_min(1.0)) * voiced
    interpolate = torch.nn.functional.interpolate
    weight = interpolate(voiced, size=length, mode="linear", align_corners=False)
    log_f0 = interpolate(log_f0, size=length, mode="linear", align_corners=False)
    log_f0 = log_f0 / weight.clamp_min(1e-6)
    gate = interpolate(voiced, size=length, mode="nearest")
    return torch.exp(log_f0) * gate

def cache_scope():
    """Context for building a tensor that outlives the call that built it.

    The holdout evaluation runs under ``torch.inference_mode()``, and any tensor
    first materialised there is an *inference tensor* -- it carries no version
    counter, so autograd refuses to save it for backward.  A per-call value is
    fine, but a value stashed on the module (a filter kernel, an attention mask)
    survives into the next training step and poisons every step after it with
    ``RuntimeError: Inference tensors cannot be saved for backward``.  The fix
    has to be at the point the cache is *filled*, not where it is used, so the
    stored tensor is a normal one whichever mode happened to see the first call.
    """

    return torch.inference_mode(False)
