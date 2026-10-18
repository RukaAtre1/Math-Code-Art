"""Per-artwork curve families: one double-rotor mechanism per piece.

The family is an artistic choice per artwork, like its palette; it is not a
genre or emotion classifier. Music still reaches the shape through the role
mapping in exhibition_v3.py, as four multipliers of the body's starting values:

    span  = R / base_R      overall extent
    inner = r1 / base_r1    length of rotor 1's arm
    fine  = r2 / base_r2    twist: phase of rotor 1's arm
    pen   = pen / base_pen  length of the pen arm: knots, star points, eddies

Every family keeps the original double rotor's three arms, ring to rotor 1,
rotor 1 to rotor 2, rotor 2 to the pen, with the long secondary arms and slow
relative speeds that give it its density. The pieces differ by mechanism:
gear ratios and arm lengths. Gear ratios are whole numbers, as on a toy
spirograph, so one cycle of s (0 to 2*pi) is the complete closed figure, k0
turns of the main arm. Every body draws with its pen; like the original
version, the screen shows the most recent figure or two clearly with bold
lines, and older strokes fade out (HOLD_FIGURES, FADE_FIGURES), optionally
leaving a faint capped trace (TRACE_ALPHA). Keeping every stroke at full
strength filled each figure into a solid blob within two minutes, because
music keeps changing arm lengths and no two passes coincide exactly.

Music never changes gear ratios; a motif (gear set, symmetry) changes only
through a slow cross-fade. The sum of arm lengths is an exact upper bound used
to keep ink inside the collider.
"""
from dataclasses import dataclass
import math

import numpy as np

# At typical pen speed the pen traces one complete figure in this time.
FIGURE_SECONDS = 6.0
# Typical filtered ink pen speed of the role mapping.
PEN_REFERENCE = 36.0
# Recent strokes are vectors: the last complete figure the pen drew stays at
# full strength and the stretch before it fades out over a quarter figure, so
# the screen always shows one coherent figure, like the original version
# (chosen 2026-10-17 over keeping every stroke; see the module docstring).
# Measured in figures, not seconds: a loud passage speeds the pen up, and a
# fixed time window then showed three or four slightly different figures.
# The newest LIVE_SECONDS carry the glowing pen head.
LIVE_SECONDS = 0.6
HOLD_FIGURES = 1.0
FADE_FIGURES = 0.25
# Faded strokes may leave a permanent trace: thinner, at TRACE_ALPHA of their
# opacity, in a raster whose opacity never exceeds TRACE_CAP, so even minutes
# of trace stay a faint pattern. TRACE_ALPHA = 0 leaves nothing.
TRACE_ALPHA = 0.0
TRACE_WIDTH = 0.45
TRACE_CAP = 0.3
# Samples per revolution of the fastest arm, before chord refinement.
SAMPLES_PER_TURN = 20
# Whole-figure rotation relative to the double rotor's ink voices; the pen's
# drawing speed is unchanged. Chosen by eye (2026-10-17): about 1.7-3.3 s per
# turn for Crystal Bloom instead of 2.5-4.9 s.
FAMILY_SPIN = 1.5
# Pen width relative to the double rotor's ink voices. Bolder widths (1.6x,
# 2x) were tried on 2026-10-17; the user preferred the original thin line
# once only the latest figure is shown.
FAMILY_WIDTH = 1.0
MORPH_SECONDS = 2.5
SWITCH_DWELL_SECONDS = 15.0
SWITCH_LEVEL = 0.7


def _terms(s, terms):
    z = np.zeros(np.shape(s), dtype=complex)
    bound = np.zeros(np.shape(s))
    for amplitude, frequency in terms:
        z = z + amplitude * np.exp(1j * frequency * s)
        bound = bound + np.abs(amplitude)
    return z, bound


def _rotor(arms):
    """Double rotor with whole-number gear ratios; motif = arm turns (k0, k1, k2).

    The main arm keeps its length (span only scales the whole figure). Rotor 1's
    arm follows `inner` and twists with `fine`; the pen arm follows `pen`, so
    accents open knots, lengthen star points or enlarge eddies. Ranges stay
    narrow enough that each mechanism keeps its character.
    """
    a0, a1, a2 = arms

    def shape(s, motif, span, inner, fine, pen):
        k0, k1, k2 = motif
        twist = np.exp(1j * 0.6 * (np.clip(fine, 0.5, 1.5) - 1.0))
        return _terms(s, [(span * a0, k0),
                          (span * a1 * np.clip(inner, 0.75, 1.3) * twist, k1),
                          (span * a2 * np.clip(0.6 + 0.4 * pen, 0.7, 1.45), k2)])
    return shape


@dataclass(frozen=True)
class Family:
    name: str
    description: str
    shape: object
    # Motif options per role, in ROLES order: rhythm, energy, timbre, pitch_class.
    motifs: tuple
    # Typical extent relative to the bound at the starting controls; sets the
    # visible size. The bound itself still caps every point in the collider.
    fill: float = 1.0

    @staticmethod
    def symmetry(motif):
        """Rotational order of the closed figure."""
        return math.gcd(*(k - motif[0] for k in motif[1:]))

    @staticmethod
    def frequency(motif):
        """Fastest arm's turns per cycle: sets the sampling density."""
        return max(abs(k) for k in motif)


FAMILIES = {
    "rotor_hub": Family(
        "rotor_hub", "Double rotor knot garland: rotor 2 divides the ring a whole number "
        "of times and the two small arms match, so every loop passes through a few bright knots",
        # k2 - k1 = m*k0 gives m knots (4, 5, 6 or 7).
        _rotor((0.72, 0.28, 0.28)),
        (((17, -38, 47), (17, -37, 65)), ((17, -39, 29), (17, -38, 47)),
         ((17, -37, 65), (19, -44, 89)), ((19, -44, 89), (17, -39, 29)))),
    "rotor_crystal": Family(
        "rotor_crystal", "Double rotor crystal star: a small first rotor (ring about 3.7 times "
        "its size) gives sharp six- and eight-point stars",
        _rotor((0.62, 0.30, 0.20)),
        # (13, -51, 37) replaced (21, -83, 61) and (19, -69, 43): with bold
        # strokes those dense eight-point sets filled into solid stars.
        (((17, -61, 71), (13, -51, 37)), ((13, -51, 37), (17, -67, 83)),
         ((17, -67, 83), (17, -63, 41)), ((17, -63, 41), (17, -61, 71)))),
    "rotor_eddy": Family(
        "rotor_eddy", "Double rotor eddy wheel: both rotors roll outside, a round rim around "
        "a ring of small eddies",
        _rotor((0.60, 0.26, 0.18)),
        (((15, 50, 92), (16, 52, 97)), ((14, 49, 89), (15, 50, 92)),
         ((16, 52, 97), (13, 43, 79)), ((13, 43, 79), (14, 49, 89)))),
    "rotor_corona": Family(
        "rotor_corona", "Double rotor corona: a slow second rotor leaves a ringed sun disc at "
        "the center inside radiating points",
        # (N, N - 8m, N + 2m): 2m-fold corona.
        _rotor((0.66, 0.30, 0.25)),
        (((19, -53, 37), (23, -65, 45)), ((17, -39, 31), (19, -53, 37)),
         ((23, -65, 45), (19, -45, 35)), ((19, -45, 35), (17, -39, 31)))),
    "rotor_waltz": Family(
        "rotor_waltz", "Double rotor waltz: slow wide gear ratios with three-fold symmetry, "
        "interlaced rotating orbits",
        _rotor((0.62, 0.34, 0.26)),
        (((11, -16, 26), (13, -20, 28)), ((10, -14, 25), (11, -16, 26)),
         ((13, -20, 28), (8, -13, 23)), ((8, -13, 23), (10, -14, 25)))),
}
# The original V3 drawing, kept so earlier deliverables stay reproducible.
LEGACY = "double_rotor"
NAMES = (LEGACY, *FAMILIES)


def evaluate(name, s, motif, span, inner, fine, pen):
    """Complex points and a per-sample upper bound on their magnitude."""
    return FAMILIES[name].shape(s, motif, span, inner, fine, pen)


def cycle_rate(name, pen_speed):
    """Radians of s per second: a full figure in FIGURE_SECONDS at typical speed."""
    return 2 * np.pi / FIGURE_SECONDS * pen_speed / PEN_REFERENCE


# analyze_track_profile() of the five exhibition recordings (2026-10-17), in
# TrackProfile field order: dynamics, bass, brightness, timbre_change, rhythm,
# harmonic_richness. Each prototype carries the family curated for that piece.
PROTOTYPES = {
    "rotor_hub": (0.75, 0.35, 0.21, 0.34, 0.34, 0.40),      # Clair de Lune
    "rotor_crystal": (0.46, 0.17, 0.59, 0.74, 0.90, 0.47),  # Sugar Plum Fairy
    "rotor_eddy": (0.32, 0.53, 0.65, 0.88, 0.53, 0.34),     # Mountain Spring
    "rotor_corona": (0.60, 0.87, 0.30, 1.00, 0.15, 0.16),   # Zarathustra: Sunrise
    "rotor_waltz": (0.40, 0.22, 0.61, 0.43, 0.68, 0.33),    # The Blue Danube
}


def recommend_family(profile):
    """Family of the most similar exhibition recording, for uncurated audio.

    Similarity is distance between whole-recording descriptors; it does not
    recognize genre, meter or emotion.
    """
    values = np.array([profile.dynamics, profile.bass, profile.brightness,
                       profile.timbre_change, profile.rhythm, profile.harmonic_richness])
    return min(PROTOTYPES, key=lambda name: float(np.linalg.norm(values - PROTOTYPES[name])))
