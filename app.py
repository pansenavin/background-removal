from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from rembg import remove
from PIL import Image, UnidentifiedImageError
import io
import numpy as np
import uvicorn
from scipy import ndimage

CORNER_PATCH = 12
UNIFORM_STD_THRESHOLD = 18
CHROMA_DISTANCE_THRESHOLD = 25
CHROMA_FEATHER = 100

# Tight threshold used only to decide connectivity to the border. Real
# photos have soft/anti-aliased or JPEG-blurred edges, so using the wider
# CHROMA_DISTANCE_THRESHOLD + CHROMA_FEATHER band for connectivity lets a
# smooth gradient "bridge" from the true background straight into an
# object of a similar light color (e.g. white pants on a white backdrop).
# Keeping this tight prevents that bridge from forming.
CONNECTIVITY_THRESHOLD = 30
EDGE_DILATION_PX = 6
ALPHA_CHOKE_THRESHOLD = 90

# A background-colored region not touching the image border is only
# treated as a real "hole" (e.g. the open loop of a necklace, the gap in a
# hoop earring) if it's fully enclosed by material that's confidently NOT
# background — i.e. far enough in color that it can't just be a bright
# highlight/fold on an object of a similar-but-not-identical shade (e.g. a
# stray light patch on off-white fabric, which is only marginally above
# CONNECTIVITY_THRESHOLD and must never be treated as a hole).
CONFIDENT_OBJECT_THRESHOLD = 80
MIN_HOLE_SIZE_PX = 15

# Exact hex color codes that belong to real objects (e.g. light-colored
# product parts) and must never be treated as background, even if close
# in color to the detected background. Add hex codes here as needed.
EXCLUDED_OBJECT_COLORS = []
EXCLUDED_COLOR_TOLERANCE = 15

# Inclusive RGB ranges (hex_low, hex_high) to protect as object colors.
# Only needed for manual overrides — border connectivity (see
# mask_border_connected_background below) already handles the common case
# of an object sharing the background's color but not touching the frame edge.
EXCLUDED_COLOR_RANGES = []


def _hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip("#")
    return tuple(int(hex_code[i:i + 2], 16) for i in (0, 2, 4))


def protect_excluded_colors(arr, alpha):
    """Forces alpha back to opaque for any pixel matching an exact hex code
    in EXCLUDED_OBJECT_COLORS, or falling inside an EXCLUDED_COLOR_RANGES
    band, so those object colors are never wiped out just because they're
    close to the background color."""
    for hex_code in EXCLUDED_OBJECT_COLORS:
        color = np.array(_hex_to_rgb(hex_code), dtype=np.int16)
        dist = np.sqrt(((arr - color) ** 2).sum(axis=2))
        alpha[dist < EXCLUDED_COLOR_TOLERANCE] = 255

    for hex_low, hex_high in EXCLUDED_COLOR_RANGES:
        low = np.array(_hex_to_rgb(hex_low), dtype=np.int16)
        high = np.array(_hex_to_rgb(hex_high), dtype=np.int16)
        in_range = np.all((arr >= low) & (arr <= high), axis=2)
        alpha[in_range] = 255

    return alpha


def detect_background_color(arr):
    """Sample the four corners; returns (is_uniform, bg_color)."""
    corners = np.concatenate([
        arr[:CORNER_PATCH, :CORNER_PATCH].reshape(-1, 3),
        arr[:CORNER_PATCH, -CORNER_PATCH:].reshape(-1, 3),
        arr[-CORNER_PATCH:, :CORNER_PATCH].reshape(-1, 3),
        arr[-CORNER_PATCH:, -CORNER_PATCH:].reshape(-1, 3),
    ]).astype(np.int16)
    is_uniform = corners.std(axis=0).mean() < UNIFORM_STD_THRESHOLD
    return is_uniform, np.median(corners, axis=0)


def decontaminate_edges(rgb: np.ndarray, alpha: np.ndarray, bg_color: np.ndarray) -> np.ndarray:
    """At partially-transparent edge pixels, the stored color is a blend of
    the object's real color and the background color (ordinary photographic
    anti-aliasing). Compositing that onto a new background lets the old
    background color show through as a fringe/halo. This un-blends it back
    toward the object's true color using the known background color."""
    alpha_frac = alpha.astype(np.float32) / 255.0
    # Below this, the pixel is mostly transparent anyway (residual tint is
    # imperceptible), and dividing by a tiny fraction would wildly amplify
    # any small deviation from a perfect linear blend (JPEG noise, etc.)
    # into a garish, wrong color instead of a clean correction.
    partial = (alpha_frac > 0.4) & (alpha_frac < 1.0)
    safe_alpha = np.clip(alpha_frac, 0.4, 1.0)[..., None]
    decontaminated = bg_color[None, None, :] + (rgb.astype(np.float32) - bg_color[None, None, :]) / safe_alpha
    decontaminated = np.clip(decontaminated, 0, 255)
    return np.where(partial[..., None], decontaminated, rgb).astype(np.uint8)


def mask_removable_background(arr: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """A background-colored pixel (tight threshold) is removable if either:
    (a) it's connected to the image border through other background-colored
    pixels (the ordinary case), or (b) it sits inside a hole fully enclosed
    by confidently-non-background material, like the open center of a
    necklace loop or a hoop earring. Case (b) deliberately excludes holes
    enclosed only by marginally-different material (e.g. a bright fold on
    off-white fabric), since those are noise, not real gaps."""
    background_core = dist < CONNECTIVITY_THRESHOLD

    labeled, _ = ndimage.label(background_core, structure=np.ones((3, 3)))
    border_labels = set(labeled[0, :]) | set(labeled[-1, :]) | set(labeled[:, 0]) | set(labeled[:, -1])
    border_labels.discard(0)
    border_connected = np.isin(labeled, list(border_labels))

    confident_object = dist > CONFIDENT_OBJECT_THRESHOLD
    filled_confident = ndimage.binary_fill_holes(confident_object)
    enclosed_by_confident_object = filled_confident & ~confident_object & background_core

    hole_labels, n_holes = ndimage.label(enclosed_by_confident_object, structure=np.ones((3, 3)))
    hole_sizes = ndimage.sum(enclosed_by_confident_object, hole_labels, range(1, n_holes + 1))
    big_enough = {i + 1 for i, size in enumerate(hole_sizes) if size >= MIN_HOLE_SIZE_PX}
    real_holes = np.isin(hole_labels, list(big_enough))

    return background_core & (border_connected | real_holes)


def chroma_key_remove(image: Image.Image) -> Image.Image:
    """Removes a solid-color background by color distance instead of
    salient-subject detection, so multiple separate items in the frame
    are all preserved rather than only the model's guessed 'main' subject."""
    arr = np.asarray(image.convert("RGB")).astype(np.int16)
    _, bg_color = detect_background_color(arr)
    dist = np.sqrt(((arr - bg_color) ** 2).sum(axis=2))

    background_removable = mask_removable_background(arr, dist)
    background_wide = dist < CHROMA_DISTANCE_THRESHOLD + CHROMA_FEATHER
    feather_zone = ndimage.binary_dilation(background_removable, iterations=EDGE_DILATION_PX) & background_wide
    removable = background_removable | feather_zone

    alpha = np.full(dist.shape, 255, dtype=np.uint8)
    feathered = np.clip((dist - CHROMA_DISTANCE_THRESHOLD) * (255 / CHROMA_FEATHER), 0, 255).astype(np.uint8)
    alpha[removable] = feathered[removable]

    # A very-low-alpha edge pixel is mostly transparent already, but its
    # stored color can still carry a visible tint from the original photo's
    # JPEG-blended boundary. Rather than trying to recover a "true" color
    # from data compression already discarded, snap it to fully transparent
    # so the faint colored haze disappears instead of lingering.
    alpha[alpha < ALPHA_CHOKE_THRESHOLD] = 0

    alpha = protect_excluded_colors(arr, alpha)
    rgb = np.asarray(image.convert("RGB"))
    rgb = decontaminate_edges(rgb, alpha, bg_color.astype(np.float32))
    rgba = np.dstack([rgb, alpha])
    return Image.fromarray(rgba, "RGBA")


def remove_background_auto(image: Image.Image) -> Image.Image:
    arr = np.asarray(image.convert("RGB"))
    is_uniform, _ = detect_background_color(arr)
    if is_uniform:
        return chroma_key_remove(image)
    return remove(image)

app = FastAPI(title="Background Removal API")
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def serve_ui():
    return FileResponse("static/index.html")

@app.post("/api/remove-bg")
def remove_background(file: UploadFile = File(...)):
    contents = file.file.read()
    try:
        input_image = Image.open(io.BytesIO(contents))
        input_image.load()
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")

    output_image = remove_background_auto(input_image)
    buf = io.BytesIO()
    output_image.save(buf, format="PNG")
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/png")

# if __name__ == "__main__":
#     uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
