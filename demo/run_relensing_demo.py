from pathlib import Path
import json, sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lenstronomy.LensModel.lens_model import LensModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SAMPLE = "lens_00932"
EXAMPLES = ROOT / "examples"
OUT = ROOT / "demo_outputs" / "relensing"
OUT.mkdir(parents=True, exist_ok=True)

def load_gray(path):
    return np.asarray(Image.open(path).convert("L"), dtype=float) / 255.0

def meta_get(meta, *keys, default=None):
    for k in keys:
        if k in meta:
            return meta[k]
    for v in meta.values():
        if isinstance(v, dict):
            found = meta_get(v, *keys, default=None)
            if found is not None:
                return found
    return default

def sample_bilinear(img, x, y):
    h, w = img.shape
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    x1 = x0 + 1; y1 = y0 + 1
    ok = (x0 >= 0) & (x1 < w) & (y0 >= 0) & (y1 < h)
    out = np.zeros_like(x, dtype=float)
    if not ok.any():
        return out
    dx = x[ok] - x0[ok]; dy = y[ok] - y0[ok]
    out[ok] = (
        img[y0[ok], x0[ok]] * (1-dx) * (1-dy) +
        img[y0[ok], x1[ok]] * dx * (1-dy) +
        img[y1[ok], x0[ok]] * (1-dx) * dy +
        img[y1[ok], x1[ok]] * dx * dy
    )
    return out

def norm(a):
    lo, hi = np.percentile(a[np.isfinite(a)], [1, 99])
    return np.clip((a - lo) / (hi - lo), 0, 1) if hi > lo else a * 0

def ncc(a, b):
    a = a.ravel() - np.mean(a)
    b = b.ravel() - np.mean(b)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 0 else 0.0

observed = load_gray(EXAMPLES / f"{SAMPLE}_observed.png")
recon = load_gray(EXAMPLES / f"{SAMPLE}_reconstructed_source64.png")
mask = load_gray(EXAMPLES / f"{SAMPLE}_true_mask.png") > 0.5
meta = json.loads((EXAMPLES / f"{SAMPLE}_metadata.json").read_text())

theta_E = float(meta_get(meta, "theta_E", "theta_e", "einstein_radius", default=1.2))
e1 = float(meta_get(meta, "e1", "lens_e1", default=0.0))
e2 = float(meta_get(meta, "e2", "lens_e2", default=0.0))
cx = float(meta_get(meta, "center_x", "lens_center_x", default=0.0))
cy = float(meta_get(meta, "center_y", "lens_center_y", default=0.0))
delta = float(meta_get(meta, "delta_pix", "deltaPix", "pixel_scale", default=0.05))

n = observed.shape[0]
row, col = np.indices(observed.shape)
theta_x = (col - (n - 1) / 2.0) * delta
theta_y = ((n - 1) / 2.0 - row) * delta

lens = LensModel(lens_model_list=["SIE"])
beta_x, beta_y = lens.ray_shooting(
    theta_x.ravel(),
    theta_y.ravel(),
    [{"theta_E": theta_E, "e1": e1, "e2": e2, "center_x": cx, "center_y": cy}],
)
beta_x = beta_x.reshape(observed.shape)
beta_y = beta_y.reshape(observed.shape)

bx = beta_x[mask]
by = beta_y[mask]
xmin, xmax = np.percentile(bx, [2, 98])
ymin, ymax = np.percentile(by, [2, 98])
xmin -= 0.2 * (xmax - xmin); xmax += 0.2 * (xmax - xmin)
ymin -= 0.2 * (ymax - ymin); ymax += 0.2 * (ymax - ymin)

h, w = recon.shape
sx = (beta_x - xmin) / (xmax - xmin) * (w - 1)
sy = (ymax - beta_y) / (ymax - ymin) * (h - 1)

relensed = sample_bilinear(recon, sx, sy)

denom = np.sum(relensed[mask] ** 2)
if denom > 0:
    relensed *= np.sum(observed[mask] * relensed[mask]) / denom

residual = observed - relensed
mse = float(np.mean((observed[mask] - relensed[mask]) ** 2))
corr = ncc(observed[mask], relensed[mask])

fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), dpi=180)

panels = [
    (observed, "Observed lens", "gray"),
    (recon, "Reconstructed source", "gray"),
    (norm(relensed), "Relensed reconstruction", "gray"),
    (residual, "Image residual", "coolwarm"),
]

for ax, (img, title, cmap) in zip(axes, panels):
    if title == "Image residual":
        lim = np.max(np.abs(img))
        ax.imshow(img, cmap=cmap, vmin=-lim, vmax=lim)
    else:
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle("Relensing / Forward Validation Demo", fontsize=14, fontweight="bold")
fig.text(
    0.5, 0.01,
    f"{SAMPLE}: reconstructed source is lensed forward and compared with observed lens | MSE={mse:.5f}, NCC={corr:.3f}",
    ha="center",
    fontsize=9,
)
fig.tight_layout(rect=[0, 0.05, 1, 0.92])

out = OUT / "Relensing_Forward_Validation_Demo.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)

print("RELENSING DEMO COMPLETE")
print("Sample:", SAMPLE)
print("Forward-image MSE:", round(mse, 5))
print("Forward-image NCC:", round(corr, 3))
print("Saved:", out)
