import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torch
from jax.scipy.signal import fftconvolve
from PIL import Image, ImageOps
from torchvision.transforms import v2
from torchvision.transforms.functional import to_pil_image
from transformers import AutoModel


def make_transform(resize_size: int = 256):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return (v2.Compose([to_tensor, to_float, normalize]), resize)


def transform_feats(x, img_size):
    x -= x.mean()
    x /= x.abs().max()
    x = jnp.asarray(
        torch.nn.functional.interpolate(
            x.unsqueeze(0).unsqueeze(0), size=(img_size, img_size), mode="bicubic"
        )[0, 0]
    )
    return x


def run() -> None:
    path = Path(sys.argv[1])
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    assert img.size[0] == img.size[1]

    ## Init.
    # Glitching input image and computing Initial Conditions via DINOv3 patch embeddings
    pretrained_model_name = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
    model = AutoModel.from_pretrained(pretrained_model_name)

    transforms = make_transform()
    inps = transforms[0](img)

    img_prc = to_pil_image(inps).resize((444, 444))
    img_prc.save(path.with_stem(path.stem.replace("_src", "_init")))

    inps = transforms[1](inps)

    with torch.inference_mode():
        outs = model(pixel_values=inps.unsqueeze(0))

    # cls = outs.last_hidden_state[:, 0]  # (B, 768)
    patches = outs.last_hidden_state[:, 1:]  # (1, 64, 768)
    feats = patches.transpose(1, 2).reshape(1, 768, 8, 8)  # (B, C, H)

    ## Sim.
    g = transform_feats(feats.mean(axis=(0, 1)), 144)
    h = transform_feats(feats.amax((0, 1)), 144)
    r = jnp.hypot(g, h) + 1e-6
    v0, w0 = g / r, h / r

    t0 = time.time()
    traj = simulate(v0, w0)
    traj = (traj - traj.min()) / (traj.max() - traj.min())
    print(f"simulate {time.time() - t0:.2f}s")

    gen_gif(
        img_prc,
        traj,
        path.with_stem(path.stem.replace("_src", "")).with_suffix(".gif"),
    )


#### 2D Reaction-Diffusion Sim.
E_, BETA, GAMMA = 0.33, 0.67, 0.33  # 67!!!
L, N, DT = 36.0, 144, 0.01
DX = L / N
T_END, SAVE_EVERY = 40.0, 8


def laplace(u):
    p = jnp.pad(u, 1, mode="edge")
    return (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4 * u) / DX**2


def step(state, _):
    v, w = state
    dv = (v - v**3 / 3 - w) / E_ + laplace(v)
    w = w + DT * E_ * (v + BETA - GAMMA * w)
    v = v + DT * dv
    return (v, w), None


def block(state, _):
    out = state[0]
    state, _ = jax.lax.scan(step, state, None, length=SAVE_EVERY)
    return state, out


@jax.jit
def simulate(v0, w0):
    n_blocks = int(round(T_END / DT)) // SAVE_EVERY
    state, frames = jax.lax.scan(block, (v0, w0), None, length=n_blocks)
    return jnp.concatenate([frames, state[0][None]], axis=0)


#### GIF Rendering
P, E, KS = 444, 60, 31  # output size, ease length (frames), Gabor kernel size


def gabor_kernel(ks, sigma, theta, lambd, gamma, psi=0.0):
    """https://en.wikipedia.org/wiki/Gabor_filter#Python"""
    r = ks // 2
    y, x = jnp.mgrid[-r : r + 1, -r : r + 1].astype(jnp.float32)
    xr = x * jnp.cos(theta) + y * jnp.sin(theta)
    yr = -x * jnp.sin(theta) + y * jnp.cos(theta)
    k = jnp.exp(-(xr**2 + gamma**2 * yr**2) / (2 * sigma**2)) * jnp.cos(
        2 * jnp.pi * xr / lambd + psi
    )
    return k - k.mean()


@jax.jit
def gabor_features(img):
    # one orientation per channel: 0, 60, 120 degrees
    r = KS // 2
    chans = []
    for c, th in enumerate(jnp.deg2rad(jnp.array([0.0, 60.0, 120.0]))):
        kernel = gabor_kernel(KS, 4.0, th, 10.0, 0.5)
        inp = jnp.pad(img[..., c], r, mode="reflect")
        out = jnp.abs(fftconvolve(inp, kernel[::-1, ::-1], mode="valid"))
        # Take top_k rather than max to ignore outliers & improve contrast
        out /= jax.lax.top_k(out.ravel(), int(out.size * 0.005) + 1)[0][-1]
        chans.append(out)
    return jnp.clip(jnp.stack(chans, -1), 0, 1)


def eased_positions(F):
    r"""Ease in-out sine is 0.5 * (1 - cos(pi * t)). \tilde{t} parametrized"""
    vel = np.ones(F + E)
    ramp = 0.5 * (1 - np.cos(np.pi * (np.arange(E) + 0.5) / E))
    vel[:E], vel[-E:] = ramp, ramp[::-1]
    x = jnp.asarray(np.concatenate([np.array([0.0]), np.cumsum(vel)]))
    return x * F / x[-1]


@jax.jit
def render(filter_traj, timeline, img_base, img_gabor):
    _traj_shape = filter_traj.shape

    t = jnp.arange(_traj_shape[0])
    filter_traj = filter_traj.reshape(_traj_shape[0], -1)
    interp_traj = jax.vmap(
        lambda traj: jnp.interp(timeline, t, traj), in_axes=1, out_axes=1
    )(filter_traj)
    interp_traj = interp_traj.reshape(-1, *_traj_shape[1:])

    interp_traj = jax.vmap(lambda a: jax.image.resize(a, (P, P), "cubic"))(interp_traj)
    interp_traj = jnp.clip(interp_traj, 0, 1)[..., None]
    return jnp.clip(
        (interp_traj * img_gabor + (1 - interp_traj) * img_base) * 255, 0, 255
    ).astype(jnp.uint8)


def gen_gif(img_base, filter_traj, out_path):
    t0 = time.time()
    img_base = jnp.asarray(np.asarray(img_base), jnp.float32) / 255
    img_gabor = gabor_features(img_base)

    timeline = eased_positions(filter_traj.shape[0] - 1)

    t1 = time.time()
    gif = np.concatenate(
        [
            np.asarray(render(filter_traj, timeline[j : j + 64], img_base, img_gabor))
            for j in range(0, len(timeline), 64)
        ]
    )
    print(f"render {time.time() - t1:.2f}s  ({len(gif)} frames)")

    # GIF encoding (PIL): one global 64-colour palette
    t2 = time.time()
    pal = Image.fromarray(np.concatenate(gif[::25], 0)).quantize(
        colors=64, method=Image.Quantize.MEDIANCUT
    )
    fwd = [
        Image.fromarray(a).quantize(palette=pal, dither=Image.Dither.NONE) for a in gif
    ]
    frames = fwd + fwd[-2:0:-1]  # forward + backward
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=50,
        loop=0,
        optimize=True,
        disposal=1,
    )
    print(
        f"encode {time.time() - t2:.2f}s  ({len(frames)} frames)  total {time.time() - t0:.2f}s"
    )
