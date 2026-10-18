"""Resonant Orbits: bounded musical palettes and original procedural scenery.

No network access or third-party image assets are used by the fallback renderer.
Local photographs/paintings require a provenance sidecar before they are used.
"""
from dataclasses import dataclass
import colorsys
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

OUTPUT_WIDTH, OUTPUT_HEIGHT, OUTPUT_DPI = 6000, 3375, 300
SERIES_TITLE = "Resonant Orbits"
ROOT = Path(__file__).resolve().parents[2]
BACKGROUND_DIR = ROOT / "02_Source_Materials" / "Backgrounds"


@dataclass(frozen=True)
class TrackConfig:
    key: str
    title: str
    pattern: str
    music: str
    composer: str
    performer: str
    license: str
    palette: tuple
    atmosphere: tuple
    layout: tuple  # x, y, scale: a hero and two supporting musical voices
    hero: int = 1
    # Curve family (curve_families.py): an artistic choice, like the palette.
    family: str = "double_rotor"
    source_url: str = ""
    note: str = "License and recording identity supplied by the project owner; source URL not yet documented."

    @property
    def background_path(self):
        return BACKGROUND_DIR / f"{self.key}.png"

    @property
    def color_profile(self):
        return COLOR_PROFILES[self.key]


TRACK_CONFIG = {
    "lunar_drift": TrackConfig(
        "lunar_drift", "Lunar Drift", "*Clair_de_lune*.ogg", "Clair de Lune",
        "Claude Debussy", "Laurens Goedhart", "CC BY 3.0",
        ("#354D73", "#50A9EE", "#A479E8", "#60D1DE", "#F6F4ED"),
        ("#090F21", "#202F4B", "#809AB5"),
        ((3.0, 2.25, 0.85), (11.35, 4.75, 2.70), (6.5, 1.7, 0.70)), family="rotor_hub"),
    "crystal_bloom": TrackConfig(
        "crystal_bloom", "Crystal Bloom", "*Dance_of_the_Sugar_Plum_Fairy*.ogg",
        "Dance of the Sugar Plum Fairy", "Pyotr Ilyich Tchaikovsky", "Kevin MacLeod",
        "CC BY 3.0", ("#68729A", "#9873EF", "#EC75B3", "#55D6B6", "#F4EDDF"),
        ("#12101F", "#2B2840", "#ADA6CD"),
        ((10.7, 4.9, 2.40), (7.5, 1.55, 0.75), (14.1, 1.55, 0.70)), hero=0, family="rotor_crystal",
        source_url="http://freemusicarchive.org/music/Kevin_MacLeod/Classical_Sampler/Dance_of_the_Sugar_Plum_Fairy",
        note="Local OGG stream tags identify Kevin MacLeod, CC BY 3.0 and this Free Music Archive source URL. Remote page not revalidated."),
    "mountain_current": TrackConfig(
        "mountain_current", "Mountain Current", "*mountain-spring*.mp3",
        "Mountain Spring (Full Version)", "kaazoom", "kaazoom", "Pixabay Content License",
        ("#294B4D", "#46BD91", "#48A5C2", "#B6CE67", "#DED1A0"),
        ("#0B181D", "#263F43", "#839C99"),
        ((3.0, 2.5, 0.95), (8.0, 4.8, 2.75), (13.1, 2.8, 0.85)), family="rotor_eddy",
        note="Owner-supplied license; Pixabay recording by kaazoom. AI-generated music, edited/mastered by kaazoom. Exact source URL not yet documented."),
    "dawn_of_an_orbit": TrackConfig(
        "dawn_of_an_orbit", "Dawn of an Orbit", "*Also_Sprach_Zarathustra*.ogg",
        "Also sprach Zarathustra: Introduction / Sunrise", "Richard Strauss", "Kevin MacLeod",
        "CC BY 3.0", ("#654951", "#ED733B", "#F7B63D", "#F4D05C", "#FFF1DD"),
        ("#080F20", "#36252A", "#C08653"),
        ((3.0, 2.8, 1.0), (9.1, 4.6, 2.90), (13.6, 2.1, 0.7)), family="rotor_corona",
        note="Local OGG stream tags identify Kevin MacLeod, Classical Sampler and CC BY 3.0. Exact recording source URL not documented."),
    "celestial_waltz": TrackConfig(
        "celestial_waltz", "Celestial Waltz", "*The_Blue_Danube*.ogg",
        "The Blue Danube / An der schönen blauen Donau", "Johann Strauss II",
        "Not documented for the selected local recording", "CC0 1.0 / Public Domain dedication",
        ("#294B83", "#488FEB", "#39C4D0", "#D8B95B", "#E9DCB7"),
        ("#080F26", "#193152", "#6D98B2"),
        ((3.0, 5.4, 0.85), (8.1, 5.7, 2.15), (13.4, 5.0, 0.85)), family="rotor_waltz"),
}

# Artistic choices, not automatically inferred emotion labels.
COLOR_PROFILES = {
    "lunar_drift": dict(mood="quiet, misty, flowing", saturation=(0.30, 0.58),
                        brightness=(0.61, 0.88), phrase_weight=0.55, color_tau=0.32),
    "crystal_bloom": dict(mood="delicate, playful, mysterious", saturation=(0.43, 0.76),
                          brightness=(0.66, 0.96), phrase_weight=0.40, color_tau=0.18),
    "mountain_current": dict(mood="clear, natural, expansive", saturation=(0.36, 0.66),
                            brightness=(0.60, 0.91), phrase_weight=0.48, color_tau=0.26),
    "dawn_of_an_orbit": dict(mood="solemn, building, monumental", saturation=(0.42, 0.80),
                            brightness=(0.55, 0.98), phrase_weight=0.60, color_tau=0.28),
    "celestial_waltz": dict(mood="elegant, turning, flowing", saturation=(0.34, 0.67),
                           brightness=(0.64, 0.93), phrase_weight=0.50, color_tau=0.24),
}


def rgb(hex_color):
    return np.array([int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)])


def resolve_audio(config, root=ROOT / "02_Source_Materials" / "Audio"):
    matches = sorted(root.glob(config.pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected one local audio file for {config.title}, found {len(matches)}")
    return matches[0]


def config_for_audio(path):
    from fnmatch import fnmatch
    return next((c for c in TRACK_CONFIG.values()
                 if fnmatch(Path(path).name.casefold(), c.pattern.casefold())), None)


def derive_palette(background_image, track_palette):
    """Use each track's vivid stops without whitening them toward the backdrop."""
    return np.array([rgb(c) for c in track_palette])


def musical_color(palette, f, role):
    """Compatibility wrapper for the unified musical stroke style."""
    color, glow, _, _ = musical_style(palette, f, role)
    return color, glow


def musical_style(palette, f, role, profile=None, drawing_mode=None):
    """Track mood + slow phrase proxies + instantaneous audio style a stroke.

    Width is in final pixels at 1280x720; callers smooth temporal changes.
    """
    p = np.asarray(palette)
    profile = profile or COLOR_PROFILES["crystal_bloom"]
    energy = np.clip(f.get("energy", 0), 0, 1)
    centroid = np.clip(f.get("timbre", 0), 0, 1)
    flux = np.clip(f.get("flux", 0), 0, 1)
    attack = np.clip(f.get("onset", 0), 0, 1)
    chroma = np.asarray(f.get("chroma", np.ones(12) / 12))
    harmonic = float(chroma @ np.sin(np.arange(12) * np.pi / 6)) / max(float(chroma.sum()), 1e-9)
    bass = np.clip(f.get("bass", energy), 0, 1)
    high = np.clip(f.get("high", centroid), 0, 1)
    beat = np.clip(f.get("beat", attack), 0, 1)
    # Each voice listens to different aspects of the same recording.
    local = (0.48*centroid + 0.30*attack + 0.22*beat,
             0.45*centroid + 0.35*energy + 0.20*(1-bass),
             0.48*centroid + 0.30*high + 0.22*flux,
             0.60*(0.5+0.5*harmonic) + 0.40*centroid)[role]
    phrase = f.get("phrase_anchor", 0.5)
    weight = profile["phrase_weight"]
    position = np.clip(weight*phrase + (1-weight)*local + 0.08*harmonic
                       + (-0.12, 0.02, 0.22, 0.10)[role], 0, 1)
    # RGB interpolation stays between intentional palette stops; HSV then
    # changes intensity without adding an unconstrained rotating hue wheel.
    location = position * 2
    lower = min(int(location), 1)
    mix = location - lower
    primary = p[1 + lower]*(1-mix) + p[2 + lower]*mix
    hue, _, _ = colorsys.rgb_to_hsv(*primary)
    phrase_energy = np.clip(f.get("phrase_energy", energy), 0, 1)
    crescendo = np.clip(f.get("crescendo", 0), -1, 1)
    sat_lo, sat_hi = profile["saturation"]
    saturation = sat_lo + (sat_hi-sat_lo)*np.clip(
        0.45*flux + 0.30*phrase_energy + 0.25*energy, 0, 1)
    val_lo, val_hi = profile["brightness"]
    value = val_lo + (val_hi-val_lo)*np.clip(
        0.45*energy + 0.25*phrase_energy + 0.25*attack + 0.05*max(0, crescendo), 0, 1)
    color = np.array(colorsys.hsv_to_rgb(hue, saturation, value))
    glow = float(0.35*energy + 0.40*attack + 0.25*flux)
    drawing_mode = drawing_mode or ("recomputed" if role < 2 else "ink")
    if drawing_mode == "recomputed":
        width = float(0.8 + 0.35*energy + 0.15*attack)
    else:
        width = float(1.4 + 0.55*energy + 0.25*attack)
    opacity = float(0.85 + 0.12*energy)
    return color, glow, width, opacity


def procedural_background(config, width, height):
    """Render quiet scenic light fields at the requested native resolution."""
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    base, haze, light = [rgb(c).astype(np.float32) for c in config.atmosphere]
    field = np.empty((height, width, 3), dtype=np.float32)
    glow = np.exp(-((x - 0.55) / 0.68)**2 - ((y - 0.63) / 0.46)**2)
    for channel in range(3):
        field[:, :, channel] = base[channel] + glow * (haze[channel] - base[channel])

    def wash(cx, cy, rx, ry, strength):
        mask = np.exp(-((x - cx) / rx)**2 - ((y - cy) / ry)**2) * strength
        for channel in range(3):
            field[:, :, channel] += mask * light[channel]

    if config.key == "lunar_drift":
        wash(0.20, 0.28, 0.17, 0.25, 0.18)
        wash(0.20, 0.79, 0.12, 0.19, 0.10)
    elif config.key == "crystal_bloom":
        wash(0.5, 0.27, 0.4, 0.43, 0.08)
        wash(0.96, 0.87, 0.26, 0.32, 0.13)
    elif config.key == "mountain_current":
        wash(0.38, 0.48, 0.68, 0.17, 0.14)
    elif config.key == "dawn_of_an_orbit":
        horizon = 0.86 + 0.15 * ((x - 0.53) / 0.8)**2
        rim = np.exp(-((y - horizon) / 0.025)**2)
        halo = np.exp(-((y - horizon) / 0.13)**2)
        for channel in range(3):
            field[:, :, channel] += light[channel] * (0.28 * rim + 0.13 * halo)
            field[:, :, channel] *= np.where(y > horizon, 0.35, 1)
        wash(0.53, 0.84, 0.17, 0.065, 0.26)
    else:
        wash(0.5, 0.73, 0.60, 0.23, 0.11)
    vignette = np.clip(1 - 0.36 * ((x - 0.5)**2 + (y - 0.5)**2) * 2, 0.65, 1)
    field *= vignette[:, :, None]
    np.clip(field, 0, 1, out=field)
    image = Image.fromarray((field * 255).astype(np.uint8))
    del field, glow
    draw = ImageDraw.Draw(image, "RGBA")
    rng = np.random.default_rng(170 + list(TRACK_CONFIG).index(config.key))

    def line(points, color, weight=1):
        draw.line([(int(px * width), int(py * height)) for px, py in points],
                  fill=color, width=max(1, round(weight * width / 1600)))

    if config.key == "lunar_drift":
        cx, cy, r = 0.20 * width, 0.27 * height, 0.029 * height
        draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill=(174, 189, 209, 140))
        for i in range(150):
            yy = rng.uniform(0.70, 0.98)
            xx = rng.normal(0.22, 0.05 + (yy - 0.7) * 0.32)
            length = rng.uniform(0.002, 0.045)
            line([(xx, yy), (xx+length, yy)], (127, 158, 188, int(rng.integers(8, 28))))
        for yy in (0.64, 0.67, 0.72):
            line([(0, yy), (1, yy + 0.005)], (88, 118, 160, 16))
    elif config.key == "mountain_current":
        for layer in range(4):
            xx = np.linspace(-0.1, 1.1, 90)
            yy = (0.49 + layer * 0.115 - 0.11 * np.sin(xx * 15 + layer * 1.8)
                  - 0.05 * np.sin(xx * 35 + layer) - 0.018 * np.sin(xx * 77))
            polygon = [(int(a*width), int(b*height)) for a, b in zip(xx, yy)]
            polygon += [(width, height), (0, height)]
            draw.polygon(polygon, fill=(24+layer*2, 49+layer, 53+layer, 75+layer*30))
        for i in range(22):
            yy = 0.67 + i * 0.014
            xx = 0.36 + 0.12 * np.sin(i * 0.13)
            line([(xx-0.012-i*0.003, yy), (xx+0.018+i*0.004, yy)], (125, 162, 157, 26), 2)
    elif config.key == "crystal_bloom":
        for side in (0, 1):
            for i in range(7):
                cx = (0.015 + i*0.024) if side == 0 else (0.985-i*0.024)
                cy = 0.84 + rng.uniform(-0.08, 0.10)
                dx, dy = rng.uniform(0.018, 0.04), rng.uniform(0.08, 0.19)
                poly = [(cx, cy-dy), (cx+dx, cy), (cx, cy+dy), (cx-dx, cy), (cx, cy-dy)]
                line(poly, (177, 180, 212, 32), 1.3)
                line([(cx, cy-dy), (cx, cy+dy)], (200, 185, 209, 24))
    elif config.key == "celestial_waltz":
        # Three quiet arcs suggest triple-time; not a claim of beat detection.
        for i in range(3):
            box = (int((0.06+i*0.025)*width), int((0.65+i*0.035)*height),
                   int((0.94-i*0.025)*width), int((1.18-i*0.01)*height))
            draw.arc(box, 184, 351, fill=(132, 165, 188, 32), width=max(1, width//1800))
        for _ in range(100):
            xx, yy = rng.uniform(0.12, 0.90), rng.uniform(0.81, 0.99)
            line([(xx, yy), (xx+rng.uniform(0.002, 0.018), yy)], (89, 145, 180, 20))
    if config.key in ("crystal_bloom", "dawn_of_an_orbit", "celestial_waltz"):
        for _ in range(85):
            xx, yy = rng.uniform(0.02, 0.98), rng.uniform(0.04, 0.8)
            if 0.28 < xx < 0.75 and 0.16 < yy < 0.79:
                continue  # keep detail out of the hero's negative space
            r = max(1, width / 4500)
            draw.ellipse((xx*width-r, yy*height-r, xx*width+r, yy*height+r),
                         fill=(190, 197, 215, int(rng.integers(20, 65))))
    return image


def prepare_background(config, width, height):
    BACKGROUND_DIR.mkdir(parents=True, exist_ok=True)
    candidates = [p for p in BACKGROUND_DIR.glob(config.key + ".*")
                  if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")]
    # Prefer a user-supplied asset over our generated fallback, deterministically.
    candidates.sort(key=lambda p: (p == config.background_path, p.name))
    if candidates:
        path = candidates[0]
        # Asset-specific sidecars avoid a .jpg and fallback .png sharing credits.
        sidecar = path.with_name(path.name + ".json")
        if not sidecar.exists():
            sidecar = path.with_suffix(".json")
        if not sidecar.exists():
            raise ValueError(f"Background provenance required: {sidecar}")
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        if meta.get("kind") == "procedural" and meta.get("generator") == "Resonant Orbits":
            with Image.open(path) as saved:
                if saved.size == (width, height):
                    return saved.convert("RGB"), dict(meta, filename=str(path))
        else:
            required = ("creator", "source_url", "license")
            if not all(meta.get(k) for k in required) or meta.get("commercial_use") is not True:
                raise ValueError(f"Incomplete commercial-use provenance: {sidecar}")
            settings = meta.get("render", {})
            with Image.open(path) as saved:
                image = ImageOps.fit(ImageOps.exif_transpose(saved).convert("RGB"), (width, height),
                                     method=Image.Resampling.LANCZOS,
                                     centering=tuple(settings.get("crop_center", (0.5, 0.5))))
            image = ImageEnhance.Color(image).enhance(settings.get("saturation", 0.72))
            image = ImageEnhance.Brightness(image).enhance(settings.get("brightness", 0.58))
            image = image.filter(ImageFilter.GaussianBlur(settings.get("blur", 0.3)*width/1600))
            a = np.asarray(image).astype(np.float32)
            x = np.linspace(-1, 1, width, dtype=np.float32)[None, :]
            y = np.linspace(-1, 1, height, dtype=np.float32)[:, None]
            a *= np.clip(1-0.12*(x*x+y*y), 0.65, 1)[:, :, None]
            # A localized soft veil protects line contrast without erasing the
            # entire photograph, its recognizable subject, or the sunrise rays.
            cx, cy = settings.get("focus_center", (0.5, 0.5))
            rx, ry = settings.get("focus_radius", (0.26, 0.40))
            veil = 1 - settings.get("focus_dim", 0.20)*np.exp(
                -(((x+1)/2-cx)/rx)**4 - (((y+1)/2-cy)/ry)**4)
            a *= veil[:, :, None]
            a *= np.array(settings.get("tint", (1, 1, 1)), dtype=np.float32)
            np.clip(a, 0, 255, out=a)
            return Image.fromarray(a.astype(np.uint8)), dict(meta, filename=str(path))
    image = procedural_background(config, width, height)
    # Cache archival-sized originals; previews must not replace an existing master.
    path = config.background_path
    if not path.exists() or (width >= OUTPUT_WIDTH and height >= OUTPUT_HEIGHT):
        image.save(path, dpi=(OUTPUT_DPI, OUTPUT_DPI))
        meta = dict(kind="procedural", generator="Resonant Orbits",
                    creator="Original procedural scenery generated by this project's code",
                    source_url="", license="Original procedural background; no third-party asset used",
                    commercial_use=True, width=width, height=height)
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    else:
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    return image, dict(meta, filename=str(path))
