"""Music-driven double rotors with continuous ink and bounded body motion.

The visual envelope stays inside a fixed collider while the music changes
the drawing and the speed of its travelling center.
"""
from collections import deque
from dataclasses import dataclass, field
import math

import librosa
import numpy as np
from .art_direction import TRACK_CONFIG, musical_style, rgb
from .curve_families import (FADE_FIGURES, FAMILIES, FAMILY_SPIN, FAMILY_WIDTH, HOLD_FIGURES, LEGACY,
                             LIVE_SECONDS, MORPH_SECONDS, NAMES, SAMPLES_PER_TURN, SWITCH_DWELL_SECONDS,
                             SWITCH_LEVEL, TRACE_ALPHA, TRACE_CAP, TRACE_WIDTH, cycle_rate, evaluate)

def multi_rotor_pen_position(R, r1, r2, d, theta, theta1, theta2):
    """Same double-rotor formula as the original notebook; isolated V3 copy."""
    x = (R-r1)*np.cos(theta)+(r1-r2)*np.cos(theta1)+d*np.cos(theta1+theta2)
    y = (R-r1)*np.sin(theta)+(r1-r2)*np.sin(theta1)+d*np.sin(theta1+theta2)
    return x, y


ROLES = ("rhythm", "energy", "timbre", "pitch_class")
COLORS = ("#F5C66B", "#78DCCB", "#EE938C", "#BCA4ED")
LABELS = ("RHYTHM / attacks + beats", "DYNAMICS / energy + bass", "TIMBRE / spectrum + change", "PITCH CLASS / chroma")
WORLD = (16.0, 9.0)
RADIUS = 1.38
PADDING = 0.18  # world units, also reserves clearance for stroke thickness
MARGIN = 0.32
TRAIL_SECONDS = 2.8
RECOMPUTED_POINTS = 2000
RECOMPUTED_TRAIL_FRACTION = 0.22
RECOMPUTED_TRAIL_STEP = 5
COLOR_HISTORY_SECONDS = 1.5
# World units between successive family samples (about 3 px at 1280x720);
# matplotlib cost grows with segment count, so stay near the double rotor's.
MAX_INK_CHORD = 0.03


def envelope(values, dt, attack, release):
    """Seconds-based attack/release filtering, independent of video FPS."""
    values = np.asarray(values, dtype=float)
    out = np.zeros_like(values)
    if not len(out):
        return out
    out[0] = values[0]
    for i in range(1, len(out)):
        tau = attack if values[i] > out[i - 1] else release
        alpha = -math.expm1(-dt / tau)
        out[i] = out[i - 1] + alpha * (values[i] - out[i - 1])
    return out


def robust_unit(values):
    values = np.nan_to_num(values)
    lo, hi = np.percentile(values, [5, 95])
    if hi - lo < 1e-9:
        return np.zeros_like(values)
    return np.clip((values - lo) / (hi - lo), 0, 1)


def phrase_features(values, chroma, dt, onset_events):
    """Slow descriptive cues, not musical phrase boundaries or emotion recognition."""
    slow_energy = envelope(values["energy"], dt, 1.8, 2.8)
    slow_timbre = envelope(values["timbre"], dt, 2.5, 2.5)
    events = np.zeros(len(slow_energy))
    events[np.asarray(onset_events, dtype=int)] = 1 / dt
    density = robust_unit(envelope(events, dt, 2.0, 2.0)) * values["activity"]
    lag = max(1, round(0.75/dt))
    past = np.concatenate([np.repeat(chroma[:, :1], lag, axis=1), chroma], axis=1)[:, :chroma.shape[1]]
    novelty = np.linalg.norm(chroma-past, axis=0)
    change = envelope(robust_unit(novelty), dt, 1.2, 2.0) * values["activity"]
    return dict(phrase_energy=slow_energy,
                phrase_anchor=np.clip(0.55*slow_timbre + 0.25*density + 0.20*change, 0, 1),
                onset_density=density, phrase_change=change,
                crescendo=np.clip((values["energy"]-slow_energy)/0.25, -1, 1))


def harmony_activity_score(chroma, gate):
    """Measure several persistent pitch classes, without naming any chord."""
    strongest, second, third = np.sort(chroma, axis=0)[-3:]
    return (np.clip((second - 0.13) / 0.10, 0, 1)
            * np.clip((third - 0.09) / 0.07, 0, 1)
            * np.clip((strongest + second + third - 0.44) / 0.24, 0, 1)
            * gate)


def sustained_harmony(score, dt, enter=0.38, leave=0.18,
                      enter_seconds=0.55, leave_seconds=1.0):
    """Gate a tonal-activity proxy so a brief note cannot summon a body."""
    active = False
    above = below = 0.0
    result = np.zeros(len(score), dtype=float)
    for i, value in enumerate(score):
        if active:
            below = below + dt if value < leave else 0.0
            if below >= leave_seconds:
                active, below = False, 0.0
        else:
            above = above + dt if value >= enter else 0.0
            if above >= enter_seconds:
                active, above = True, 0.0
        result[i] = float(active)
    return result


@dataclass(frozen=True)
class TrackProfile:
    dynamics: float
    bass: float
    brightness: float
    timbre_change: float
    rhythm: float
    harmonic_richness: float


def analyze_track_profile(path):
    """Use the whole recording to choose geometry, even for short previews."""
    sr, hop, n_fft = 11025, 512, 2048
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    spectrum = abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    rms = librosa.feature.rms(S=spectrum, frame_length=n_fft)[0]
    db = librosa.amplitude_to_db(rms, ref=max(float(rms.max()), 1e-8), top_db=80)
    centroid = librosa.feature.spectral_centroid(S=spectrum, sr=sr)[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    peaks = librosa.onset.onset_detect(onset_envelope=onset, sr=sr, hop_length=hop)
    onset_standard = (onset - onset.mean()) / (onset.std() + 1e-8)
    lags = range(round(0.25 * sr / hop), round(1.1 * sr / hop))
    regularity = max(float(np.mean(onset_standard[:-lag] * onset_standard[lag:]))
                     for lag in lags)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    bass_ratio = (spectrum[freqs < 250].sum(axis=0)
                  / np.maximum(spectrum.sum(axis=0), 1e-8))
    chroma = librosa.feature.chroma_stft(S=spectrum ** 2, sr=sr, hop_length=hop)
    chroma /= np.maximum(chroma.sum(axis=0), 1e-8)
    harmony = harmony_activity_score(chroma, np.ones(chroma.shape[1]))
    unit = lambda value: float(np.clip(value, 0.0, 1.0))
    return TrackProfile(
        dynamics=unit((np.percentile(db, 90) - np.percentile(db, 10)) / 30),
        bass=unit(np.median(bass_ratio) / 0.5),
        brightness=unit((np.median(centroid) - 450) / 1200),
        timbre_change=unit((np.percentile(centroid, 90)
                            - np.percentile(centroid, 10)) / 1200),
        rhythm=unit(regularity / 0.6) * unit(len(peaks) / (len(y) / sr) / 2.8),
        harmonic_richness=unit(float(np.mean(harmony))),
    )


class Features:
    """Analyze the whole recording so previews share one feature timeline.

    Onset is an attack measure, not a promise of beat synchronization.
    Spectral centroid is a brightness proxy, not instrument identification.
    Optional chroma is a pitch-class distribution, not chord recognition.
    """
    def __init__(self, path, start=0.0, duration=None):
        sr, hop = 22050, 256
        y, _ = librosa.load(str(path), sr=sr)
        full_duration = len(y) / sr
        if start < 0 or start >= full_duration:
            raise ValueError("Start must be within the audio recording.")
        self.duration = min(full_duration - start,
                            full_duration if duration is None else duration)
        if self.duration < 0.1:
            raise ValueError("Audio excerpt must contain at least 0.1 seconds.")
        self.start = start
        self.sr = sr
        spectrum = abs(librosa.stft(y, n_fft=2048, hop_length=hop))
        rms = librosa.feature.rms(S=spectrum)[0]
        centroid = librosa.feature.spectral_centroid(S=spectrum, sr=sr)[0]
        bandwidth = librosa.feature.spectral_bandwidth(S=spectrum, sr=sr)[0]
        onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        flux = np.sum(np.maximum(np.diff(spectrum, axis=1, prepend=spectrum[:, :1]), 0), axis=0)
        frequencies = librosa.fft_frequencies(sr=sr, n_fft=2048)
        bass = np.sqrt(np.mean(spectrum[frequencies < 250] ** 2, axis=0))
        mid = np.sqrt(np.mean(spectrum[(frequencies >= 250) & (frequencies < 2500)] ** 2, axis=0))
        high = np.sqrt(np.mean(spectrum[frequencies >= 2500] ** 2, axis=0))
        tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset, sr=sr, hop_length=hop)
        self.tempo = float(np.clip(float(np.atleast_1d(tempo)[0]) / 120, 0.5, 1.8))
        beat = np.zeros_like(onset)
        for frame in beat_frames:
            lo, hi = max(0, int(frame)-5), min(len(beat), int(frame)+17)
            offsets = (np.arange(lo, hi)-frame)*hop/sr
            beat[lo:hi] = np.maximum(beat[lo:hi], np.exp(-0.5*(offsets/0.065)**2))
        # Gate feature noise in silence; a silent frame must not look energetic.
        gate = np.clip(rms / max(float(rms.max()) * 0.025, 1e-8), 0, 1)
        rms_db = librosa.amplitude_to_db(rms, ref=max(float(rms.max()), 1e-8), top_db=60)
        dt = hop / sr
        self.values = {
            "onset": envelope(robust_unit(onset) * gate, dt, 0.025, 0.16),
            "energy": envelope(robust_unit(rms_db) * gate, dt, 0.16, 0.65),
            "timbre": envelope(robust_unit(centroid) * gate, dt, 0.35, 0.65),
            "bandwidth": envelope(robust_unit(bandwidth) * gate, dt, 0.20, 0.45),
            "flux": envelope(robust_unit(flux) * gate, dt, 0.04, 0.22),
            "bass": envelope(robust_unit(bass) * gate, dt, 0.08, 0.35),
            "mid": envelope(robust_unit(mid) * gate, dt, 0.15, 0.40),
            "high": envelope(robust_unit(high) * gate, dt, 0.08, 0.35),
            "beat": envelope(beat * gate, dt, 0.015, 0.11),
        }
        self.times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
        chroma = librosa.feature.chroma_stft(S=spectrum ** 2, sr=sr, hop_length=hop)
        chroma /= np.maximum(chroma.sum(axis=0), 1e-8)
        # This is a conservative polyphonic tonal-activity cue, not chord
        # recognition: several persistent pitch classes must share the energy.
        chroma_fast = np.stack([envelope(c, dt, 0.12, 0.20) for c in chroma])
        harmony_score = harmony_activity_score(chroma_fast, gate)
        harmony_score = envelope(harmony_score, dt, 0.16, 0.22)
        self.values["harmony_score"] = harmony_score
        # Relative thresholds make the extra voice mark the harmonically rich
        # passages of this recording, rather than occupy nearly every frame.
        enter = float(np.clip(np.percentile(harmony_score, 80), 0.55, 0.85))
        leave = float(np.clip(np.percentile(harmony_score, 45), 0.22, 0.55))
        self.values["harmony_active"] = sustained_harmony(
            harmony_score, dt, enter=enter, leave=leave,
            enter_seconds=0.5, leave_seconds=0.75)
        self.chroma = np.stack([envelope(c, dt, 0.7, 0.7) for c in chroma])
        self.values["activity"] = envelope(gate, dt, 0.1, 0.4)
        events = librosa.onset.onset_detect(onset_envelope=onset, sr=sr, hop_length=hop)
        self.values.update(phrase_features(self.values, self.chroma, dt, events))

    def sample(self, t):
        absolute_time = t + self.start
        result = {k: float(np.interp(absolute_time, self.times, v))
                  for k, v in self.values.items()}
        result["chroma"] = np.array([np.interp(absolute_time, self.times, c)
                                     for c in self.chroma])
        result["tempo"] = self.tempo
        return result


@dataclass
class Body:
    role: str
    index: int
    x: float
    y: float
    vx: float
    vy: float
    radius: float = RADIUS
    phase: float = 0.0
    hue: float = 0.74
    size: float = 0.85
    pen: float = 0.32
    linewidth: float = 1.1
    alpha: float = 0.85
    R: float = 1.25
    r1: float = 0.53
    r2: float = 0.247
    pen_speed: float = 20.0
    theta: float = 0.0
    theta1: float = 0.0
    theta2: float = 0.0
    elapsed: float = 0.0
    trail_seconds: float = TRAIL_SECONDS
    angle: float = 0.0
    angular_speed: float = 0.5
    direction: float = 1.0
    display_scale: float = 0.94
    base_R: float = 1.25
    base_r1: float = 0.53
    base_r2: float = 0.247
    base_pen: float = 0.32
    base_hue: float = 0.12
    wobble_frequency: float = 0.27
    wobble_phase: float = 0.0
    motion_speed: float = 0.8
    drawing_mode: str = "ink"
    trail_offset: int = 0
    presence: float = 1.0
    palette: object = None
    paint_rgb: object = None
    glow: float = 0.0
    color_profile: object = None
    # Curve family (see curve_families.py). The legacy double rotor keeps the
    # original R/r1/r2/d drawing; other families read the same four values as
    # multipliers of their starting geometry, biased by the whole recording.
    family: str = LEGACY
    shape_bias: object = None
    motif_index: int = 0
    motif_target: object = None
    morph: float = 0.0
    last_switch: float = 0.0
    switches: int = 0
    cycle: float = 0.0
    # Family ink that left the live window: (n, 9) arrays waiting to be
    # rasterized. Each chunk starts with the previous chunk's last sample, so
    # chunks join into one unbroken pen path. Nothing is ever discarded.
    settled: list = field(default_factory=list, repr=False)
    settled_tail: object = None
    style_history: deque = field(default_factory=deque, repr=False)
    # Birth time, x, y, line width, opacity, legacy hue, curated R/G/B and,
    # for families, the figure phase s.
    # Stored in the body's local
    # coordinates: ink moves with its collider but is NEVER re-shaped later.
    ink: deque = field(default_factory=deque, repr=False)

    def geometry(self):
        return np.array([self.R, self.r1, self.r2, self.pen, self.size, self.pen_speed])

    def shape_controls(self, R, r1, r2, pen):
        """Span, inner, fine and pen multipliers for a curve family."""
        bias = np.ones(4) if self.shape_bias is None else self.shape_bias
        return (R/self.base_R*bias[0], r1/self.base_r1*bias[1],
                r2/self.base_r2*bias[2], pen/self.base_pen*bias[3])

    def motif_frequency(self):
        family = FAMILIES[self.family]
        motifs = family.motifs[self.index]
        active = [self.motif_index] + ([] if self.motif_target is None else [self.motif_target])
        return max(family.frequency(motifs[i]) for i in active)

    def family_points(self, s, controls, morph):
        """Points, magnitude bound and reference bound, cross-fading motifs."""
        motifs = FAMILIES[self.family].motifs[self.index]
        fill = FAMILIES[self.family].fill
        bias = np.ones(4) if self.shape_bias is None else self.shape_bias

        def one(motif):
            z, bound = evaluate(self.family, s, motif, *controls)
            return z, bound, fill*float(evaluate(self.family, 0.0, motif, *bias)[1])

        z, bound, reference = one(motifs[self.motif_index])
        if self.motif_target is not None:
            # Integer-frequency closed curves blend into closed curves, and the
            # blended bound still bounds the blend (triangle inequality).
            w = np.clip(morph, 0.0, 1.0)
            w = w*w*(3-2*w)
            z2, bound2, reference2 = one(motifs[self.motif_target])
            z = (1-w)*z + w*z2
            bound = (1-w)*bound + w*bound2
            reference = (1-w)*reference + w*reference2
        return z, bound, reference

    def advance_motif(self, f, dt):
        """Slow motif changes; returns the cross-fade position at frame start.

        A switch needs a strong slow chroma-change cue and a long dwell, so the
        lobe count changes with larger musical sections, not single notes.
        """
        if self.motif_target is not None and self.morph >= 1.0:
            self.motif_index, self.motif_target, self.morph = self.motif_target, None, 0.0
        start = self.morph
        options = len(FAMILIES[self.family].motifs[self.index])
        if self.motif_target is not None:
            self.morph = min(1.0, self.morph + dt/MORPH_SECONDS)
        elif (options > 1 and self.elapsed - self.last_switch >= SWITCH_DWELL_SECONDS
              and f.get("phrase_change", 0.0) >= SWITCH_LEVEL):
            self.motif_target = (self.motif_index + 1) % options
            self.morph = min(1.0, dt/MORPH_SECONDS)
            self.last_switch = self.elapsed
            self.switches += 1
        return start

    def draw_family_ink(self, previous, current, morph_start, dt, old_width, old_alpha, old_hue, old_rgb):
        """Integrate the cycle phase so a parameter change never teleports the pen."""
        speed = cycle_rate(self.family, max(abs(previous[-1]), abs(current[-1])))
        steps = max(1, math.ceil(dt * 600),
                    math.ceil(speed*dt*self.motif_frequency()*SAMPLES_PER_TURN/(2*math.pi)))

        def sample(steps):
            mix = np.arange(1, steps + 1) / steps
            R, r1, r2, d, size, pen_speed = previous[:, None] + (current - previous)[:, None]*mix
            s = self.cycle + np.cumsum(cycle_rate(self.family, pen_speed)*dt/steps)
            controls = self.shape_controls(R, r1, r2, d)
            z, bound, reference = self.family_points(s, controls, morph_start + mix*(self.morph - morph_start))
            # Same collider rule as the double rotor: size sets the visible extent,
            # the exact bound caps it inside the fixed circle.
            scale = (self.radius - PADDING)*np.minimum(size/reference, 1.0/bound)
            return mix, scale*z, s

        mix, points, phase = sample(steps)
        # Rays and curls can sweep fast; resample until no chord is visible.
        last = complex(*self.ink[-1][1:3]) if self.ink else points[0]
        chord = float(np.max(np.abs(np.diff(points, prepend=last))))
        if chord > MAX_INK_CHORD:
            mix, points, phase = sample(min(steps*math.ceil(chord/MAX_INK_CHORD), 4000))
        px, py = points.real, points.imag
        self.cycle = float(phase[-1])
        hue_delta = (self.hue - old_hue + 0.5) % 1.0 - 0.5
        for i, m in enumerate(mix):
            paint = old_rgb + m*(self.paint_rgb - old_rgb)
            self.ink.append((self.elapsed + m*dt, px[i], py[i],
                             old_width + m*(self.linewidth - old_width),
                             old_alpha + m*(self.alpha - old_alpha),
                             (old_hue + m*hue_delta) % 1.0, *paint, phase[i]))

    def update_visual(self, f, dt):
        previous = self.geometry()
        if self.palette is None:
            self.palette = np.array([rgb(c) for c in TRACK_CONFIG["lunar_drift"].palette])
        if self.paint_rgb is None:
            self.paint_rgb = np.asarray(self.palette[1]).copy()
        old_rgb = self.paint_rgb.copy()
        profile = self.color_profile or TRACK_CONFIG["crystal_bloom"].color_profile
        color, glow, stroke_width, stroke_alpha = musical_style(
            self.palette, f, self.index, profile, self.drawing_mode)
        if self.family != LEGACY:
            stroke_width *= FAMILY_WIDTH
        self.paint_rgb +=(color - self.paint_rgb) * (-math.expm1(-dt / profile["color_tau"]))
        self.glow += (glow - self.glow) * (-math.expm1(-dt / 0.18))
        old_width, old_alpha, old_hue = self.linewidth, self.alpha, self.hue
        onset, energy, bright = f["onset"], f["energy"], f["timbre"]
        beat, bass = f.get("beat", 0.0), f.get("bass", 0.0)
        flux, bandwidth = f.get("flux", 0.0), f.get("bandwidth", 0.0)
        tempo = f.get("tempo", 1.0)
        wob1 = math.sin(2*math.pi*self.wobble_frequency*self.elapsed+self.wobble_phase)
        wob2 = math.cos(2*math.pi*self.wobble_frequency*1.31*self.elapsed+self.wobble_phase)
        accent = min(1.0, 0.70*onset+0.55*beat+0.15*flux)
        # Primary drivers shape the identity; secondary drivers add motion and
        # variation. This deliberately restores broader mappings from the old
        # notebook without giving all bodies the same global scale or velocity.
        if self.role == "rhythm":
            target = np.array([self.base_R*(0.86+0.18*accent+0.13*bright),
                self.base_r1*(0.90+0.18*accent+0.07*wob1),
                self.base_r2*(1.03-0.32*accent+0.07*wob2),
                self.base_pen*(0.65+1.45*accent+0.35*flux+0.30*energy),
                0.78+0.17*energy, 12.0+22.0*accent+5.0*tempo])
            spin = self.direction*(0.42+0.50*tempo+1.65*accent)
            self.hue = (self.base_hue+0.08*accent+0.02*flux) % 1
        elif self.role == "energy":
            target = np.array([self.base_R*(0.78+0.38*energy+0.12*bright),
                self.base_r1*(0.84+0.27*bass+0.08*bandwidth+0.09*wob1),
                self.base_r2*(0.82+0.24*energy+0.12*beat+0.12*wob2),
                self.base_pen*(0.40+1.75*energy+0.4*bass+0.25*onset),
                0.64+0.32*energy, 15.0+12.0*energy+5.0*bass])
            spin = self.direction*(0.35+1.00*energy+0.35*bass)
            self.hue = (self.base_hue+0.11*energy-0.03*bass) % 1
        elif self.role == "timbre":
            wobble = 0.04+0.16*bandwidth
            target = np.array([self.base_R*(0.84+0.33*bright),
                self.base_r1*(0.76+0.40*bright+wobble*wob1),
                self.base_r2*(0.74+0.39*bandwidth+wobble*wob2),
                self.base_pen*(0.55+1.55*bright+0.45*flux),
                0.88, 15.0+13.0*bandwidth+10.0*flux])
            spin = self.direction*(0.36+0.85*bright+0.65*flux)
            self.hue = (self.base_hue-0.12*bright+0.04*bandwidth) % 1
        else:
            c = f["chroma"]
            # Confidence hold avoids hue flipping between nearly equal peaks.
            order = np.argsort(c)
            if f.get("activity", 1.0) > 0.15 and c[order[-1]] - c[order[-2]] > 0.025:
                target = float(order[-1]) / 12
                delta = (target - self.hue + 0.5) % 1.0 - 0.5
                self.hue = (self.hue + delta * (-math.expm1(-dt / 0.6))) % 1.0
            spread = float(1.0 - np.max(c))
            target = np.array([self.base_R*(0.95+0.15*spread),
                self.base_r1*(0.88+0.18*spread+0.06*wob1),
                self.base_r2*(0.90+0.20*bright+0.06*wob2),
                self.base_pen*(0.75+0.55*spread), 0.86, 18+6*spread])
            spin = self.direction*(0.45+0.60*spread)
        # All voices share the audio-driven style; rotor geometry is unchanged.
        tau = 0.055 if stroke_width > old_width else 0.22
        self.linewidth = old_width + (stroke_width-old_width)*(-math.expm1(-dt/tau))
        self.alpha = old_alpha + (stroke_alpha-old_alpha)*(-math.expm1(-dt/0.12))
        if self.drawing_mode == "recomputed":
            if not self.style_history:
                self.style_history.append((self.elapsed, old_width, old_alpha, *old_rgb))
            self.style_history.append((self.elapsed+dt, self.linewidth, self.alpha, *self.paint_rgb))
            while len(self.style_history) > 2 and self.style_history[1][0] < self.elapsed+dt-COLOR_HISTORY_SECONDS:
                self.style_history.popleft()
        # Attacks shape the next few pen samples quickly; the visible history
        # still provides continuity instead of requiring heavy parameter lag.
        if self.drawing_mode == "recomputed":
            # Small ratio changes shift the late turns of a 40-turn curve a
            # long way. Let the structure settle while accents stay quick.
            times = np.array([0.75, 1.1, 1.1, 0.28, 0.38, 0.12])
            response = -np.expm1(-dt / times)
            spin *= 0.72
        else:
            response = -math.expm1(-dt / 0.075)
            spin *= 1.45 * (FAMILY_SPIN if self.family != LEGACY else 1.0)
            target[-1] *= 1.55
        filtered = previous+(target-previous)*response
        self.R, self.r1, self.r2, self.pen, self.size, self.pen_speed = filtered
        self.angular_speed += (spin-self.angular_speed)*(-math.expm1(-dt/0.08))
        # Whole accumulated drawing rotates together, separately from the pen.
        self.angle = (self.angle+self.angular_speed*dt) % (2*math.pi)
        self.display_scale = 0.96 if self.role != "energy" else 0.83+0.15*energy
        if self.role == "pitch_class" and self.presence <= 0:
            self.release_ink()
            self.elapsed += dt
            return
        morph_start = self.advance_motif(f, dt) if self.family != LEGACY else 0.0
        if self.drawing_mode == "recomputed":
            # The old renderer redraws a fresh musical curve at every frame.
            # There is no persistent local ink to integrate for these bodies.
            self.elapsed += dt
            return
        current = self.geometry()
        if self.family != LEGACY:
            self.draw_family_ink(previous, current, morph_start, dt,
                                 old_width, old_alpha, old_hue, old_rgb)
        else:
            # Several pen samples per video frame, with interpolated radii. Integrate
            # the rotor phases instead of multiplying changing ratios by total time:
            # that multiplication would teleport the pen whenever the music changes.
            # 600 samples/s is the baseline. Fast pen motion gets more samples so
            # the nested rotor path stays smooth rather than forming sharp chords.
            max_speed = max(abs(previous[-1]), abs(current[-1]))
            max_ratio = max(self.R / self.r1, self.r1 / self.r2)
            steps = max(1, math.ceil(dt * 600), math.ceil(dt * max_speed * max_ratio / 0.10))
            sub_dt = dt / steps
            reference_bound = (abs(self.base_R - self.base_r1)
                               + abs(self.base_r1 - self.base_r2) + abs(self.base_pen))
            for step in range(1, steps + 1):
                mix = step / steps
                R, r1, r2, d, size, speed = previous + (current - previous) * mix
                angle = speed * sub_dt
                inner = -(R / r1) * angle
                self.theta += angle
                self.theta1 += inner
                self.theta2 += -(r1 / r2) * inner
                x, y = multi_rotor_pen_position(R, r1, r2, d, self.theta, self.theta1, self.theta2)
                bound = abs(R - r1) + abs(r1 - r2) + abs(d)
                # Use a fixed reference envelope so changes in the radii and pen
                # distance can change the visible size. Cap each new point against
                # the fixed collider; recorded ink must never leave its boundary.
                scale = (self.radius - PADDING) * min(size / reference_bound, 1.0 / bound)
                # Store ink in body-local coordinates. Rigid rotation happens only
                # in world_points(), so OLD and NEW ink really rotate together.
                px, py = scale*x, scale*y
                width = old_width + mix * (self.linewidth - old_width)
                alpha = old_alpha + mix * (self.alpha - old_alpha)
                hue_delta = (self.hue - old_hue + 0.5) % 1.0 - 0.5
                hue = (old_hue + mix * hue_delta) % 1.0
                paint = old_rgb + mix * (self.paint_rgb - old_rgb)
                self.ink.append((self.elapsed + step * sub_dt, px, py, width, alpha, hue, *paint))
        self.elapsed += dt
        if self.family != LEGACY:
            # Keep the figure just drawn plus the fading stretch before it.
            self.settle(self.cycle - 2*math.pi*(HOLD_FIGURES + FADE_FIGURES), column=9)
        else:
            while self.ink and self.ink[0][0] < self.elapsed - self.trail_seconds:
                self.ink.popleft()

    def settle(self, cutoff, column=0):
        """Move family ink older than cutoff (time, or figure phase in column 9)
        into the settled record that feeds the trace layer."""
        moved = []
        while self.ink and self.ink[0][column] < cutoff:
            moved.append(self.ink.popleft())
        if moved:
            head = [] if self.settled_tail is None else [self.settled_tail]
            self.settled.append(np.asarray(head + moved))
            self.settled_tail = moved[-1]

    def drain_settled(self):
        """Hand settled chunks to the renderer's raster layer exactly once."""
        chunks, self.settled = self.settled, []
        return chunks

    def release_ink(self):
        """A hidden body keeps family strokes; legacy ink is simply cleared."""
        if self.family != LEGACY:
            self.settle(math.inf)
        else:
            self.ink.clear()

    def update_motion(self, f, dt):
        """Let musical accents accelerate the center without changing its collider."""
        onset = f["onset"]
        energy = f["energy"]
        beat = f.get("beat", 0.0)
        flux = f.get("flux", 0.0)
        bass = f.get("bass", 0.0)
        if self.role == "rhythm":
            drive = min(1.0, 0.65*onset + 0.70*beat + 0.15*flux)
        elif self.role == "energy":
            drive = min(1.0, 0.75*energy + 0.35*bass)
        elif self.role == "timbre":
            drive = min(1.0, 0.40*energy + 0.60*flux)
        else:
            drive = min(1.0, 0.65*energy + 0.35*beat)
        target_speed = 0.68 + 1.55*drive
        tau = 0.055 if target_speed > self.motion_speed else 0.32
        self.motion_speed += (target_speed - self.motion_speed) * (-math.expm1(-dt/tau))
        current_speed = math.hypot(self.vx, self.vy)
        if current_speed > 1e-9:
            scale = self.motion_speed/current_speed
            self.vx *= scale
            self.vy *= scale

    def update_presence(self, enabled, dt):
        """Fade a temporary harmonic voice in and out without flickering."""
        if enabled:
            self.presence = min(1.0, self.presence + dt / 0.45)
        else:
            self.presence = max(0.0, self.presence - dt / 0.8)
            if self.presence == 0.0:
                self.release_ink()

    def local_points(self):
        if not self.ink:
            return np.array([]), np.array([])
        points = np.asarray(self.ink)
        return points[:, 1], points[:, 2]

    def world_points(self, points):
        c, s = math.cos(self.angle), math.sin(self.angle)
        rotation = np.array([[c, -s], [s, c]])
        return (points @ rotation.T)*self.display_scale + np.array([self.x, self.y])

    def recomputed_style(self, color_positions, count):
        """Colors and widths along a recomputed curve, tail (0) to head (1)."""
        # Geometry is still regenerated from this frame's parameters.
        # Its tail-to-head color maps actual recent audio styles to a
        # virtual 1.5-second timeline; it is not historical geometry.
        if self.style_history:
            history = np.asarray(self.style_history)
            begin = max(history[0, 0], self.elapsed-COLOR_HISTORY_SECONDS)
            times = begin + color_positions*(self.elapsed-begin)
            style = np.column_stack([np.interp(times, history[:, 0], history[:, k]) for k in range(1, 6)])
            return np.column_stack([style[:, 2:5], style[:, 1]]), style[:, 0]
        paint = self.paint_rgb if self.paint_rgb is not None else [0.67, 0.79, 0.88]
        return np.tile([*paint, self.alpha], (count, 1)), np.full(count, self.linewidth)

    def head_count(self):
        """Newest stroke segments (younger than LIVE_SECONDS): the glowing head."""
        count = 0
        for sample in reversed(self.ink):
            if self.elapsed - sample[0] >= LIVE_SECONDS:
                break
            count += 1
        return count

    def strokes(self, frame_idx=0, detail=1):
        """Return the current sliding curve, the recent ink or a family's live head."""

        if self.drawing_mode == "recomputed":
            t = np.linspace(0.0, 40 * 2 * np.pi, RECOMPUTED_POINTS)
            theta1 = -(self.R / self.r1) * t
            theta2 = -(self.r1 / self.r2) * theta1
            x, y = multi_rotor_pen_position(self.R, self.r1, self.r2, self.pen,
                                             t, theta1, theta2)
            bound = abs(self.R - self.r1) + abs(self.r1 - self.r2) + abs(self.pen)
            reference_bound = (abs(self.base_R - self.base_r1)
                               + abs(self.base_r1 - self.base_r2) + abs(self.base_pen))
            scale = (self.radius - PADDING) * min(self.size/reference_bound, 1.0/bound)
            points = np.column_stack([x, y]) * scale
            start = (self.trail_offset + frame_idx*RECOMPUTED_TRAIL_STEP) % RECOMPUTED_POINTS
            count = int(RECOMPUTED_POINTS * RECOMPUTED_TRAIL_FRACTION)
            indices = (start + np.arange(count)) % RECOMPUTED_POINTS
            xy = self.world_points(points[indices])
            segments = np.stack([xy[:-1], xy[1:]], axis=1)
            color_positions = (np.arange(count-1)+0.5)/(count-1)
            if detail > 1:
                # Tessellate the SAME analytic trajectory between the existing
                # samples for print. Do not smooth/rewrite musical parameters.
                # A modulo wrap is a pen lift, not a chord across the drawing.
                first, last = indices[:-1], indices[1:]
                continuous = last == first + 1
                tt = t[first[continuous], None] + (t[last[continuous]]-t[first[continuous]])[:, None] * np.linspace(0, 1, detail+1)
                a = -(self.R/self.r1)*tt
                b = -(self.r1/self.r2)*a
                xx, yy = multi_rotor_pen_position(self.R, self.r1, self.r2, self.pen, tt, a, b)
                smooth = self.world_points(np.stack([xx, yy], axis=-1)*scale)
                segments = np.stack([smooth[:, :-1], smooth[:, 1:]], axis=2).reshape(-1, 2, 2)
                color_positions = ((np.arange(count-1)[continuous, None]
                                    + (np.arange(detail)+0.5)[None, :]/detail)/(count-1)).ravel()
            return (segments, *self.recomputed_style(color_positions, len(segments)))

        live = list(self.ink)
        if self.family != LEGACY and self.settled_tail is not None:
            live.insert(0, self.settled_tail)
        if len(live) < 2:
            return np.empty((0, 2, 2)), np.empty((0, 4)), np.empty(0)
        history = np.asarray(live)
        xy = self.world_points(history[:, 1:3])
        segments = np.stack([xy[:-1], xy[1:]], axis=1)
        age = self.elapsed - history[1:, 0]
        if self.family != LEGACY:
            # Full strength for the last HOLD_FIGURES of pen travel (the figure
            # just drawn), then a smooth fade that reaches exactly the trace
            # style when the stroke settles into the raster.
            figures = (self.cycle - history[1:, 9]) / (2*math.pi)
            u = np.clip((figures - HOLD_FIGURES) / FADE_FIGURES, 0.0, 1.0)
            u = u*u*(3 - 2*u)
            alpha = history[1:, 4] * (1 - u*(1 - TRACE_ALPHA))
            widths = history[1:, 3] * (1 - u*(1 - TRACE_WIDTH))
            return segments, np.column_stack([history[1:, 6:9], alpha]), widths
        # A short, bright head makes each new musical gesture readable.
        fade = np.clip((self.trail_seconds - age) / 0.55, 0.0, 1.0)
        alpha = history[1:, 4] * (0.18 + 0.82 * np.exp(-age / 1.1)) * fade
        return segments, np.column_stack([history[1:, 6:9], alpha]), history[1:, 3]


def make_bodies(count=3, seed=17, track_profile=None, palette=None, color_profile=None,
                family=LEGACY):
    if count not in (3, 4):
        raise ValueError("Choose three or four spirographs.")
    if family not in NAMES:
        raise ValueError(f"Unknown curve family {family!r}; choose from {', '.join(NAMES)}.")
    rng = np.random.default_rng(seed)
    positions = [(3.1, 3.0), (8.0, 6.0), (12.9, 3.0)] if count == 3 else [
        (4.0, 2.6), (12.0, 2.6), (4.0, 6.4), (12.0, 6.4)]
    bodies = []
    for i, (x, y) in enumerate(positions):
        angle = rng.uniform(0, 2 * math.pi)
        speed = 0.8
        body = Body(ROLES[i], i, x, y, speed * math.cos(angle), speed * math.sin(angle))
        body.color_profile = dict(color_profile) if color_profile else None
        if palette is not None:
            body.palette = np.asarray(palette).copy()
        # Different double-rotor proportions, within the original model's range.
        default = np.array((
            (1.25, 0.53, 0.247, 0.28), (1.18, 0.47, 0.213, 0.30),
            (1.30, 0.52, 0.251, 0.34), (1.21, 0.49, 0.229, 0.32))[i])
        profile = default * rng.uniform(0.90, 1.10, 4)
        if track_profile is not None:
            p = track_profile
            if i == 0:  # rhythm: active recordings start with clearer lobes
                factors = (0.90+0.22*p.rhythm, 1.10-0.18*p.rhythm,
                           0.92+0.16*p.brightness, 0.83+0.30*p.rhythm)
            elif i == 1:  # dynamics: bass and long swells set the envelope
                factors = (0.88+0.24*p.dynamics, 0.92+0.16*p.bass,
                           1.08-0.18*p.bass, 0.80+0.42*p.bass)
            elif i == 2:  # timbre: spectral spread sets rotor proportions
                factors = (0.90+0.20*p.timbre_change,
                           0.88+0.24*p.brightness,
                           1.10-0.18*p.timbre_change,
                           0.84+0.32*p.brightness)
            else:  # harmony: tonal density gives the temporary voice identity
                factors = (0.92+0.18*p.harmonic_richness,
                           1.06-0.14*p.harmonic_richness,
                           0.90+0.18*p.timbre_change,
                           0.86+0.24*p.harmonic_richness)
            profile *= factors
        body.R, body.r1, body.r2, body.pen = profile
        body.base_R, body.base_r1, body.base_r2, body.base_pen = profile
        # Families read the seed and whole-recording factors as a shape bias,
        # so the same piece and seed always start from the same figure.
        body.family = family
        body.shape_bias = profile / default
        body.base_hue = (0.10, 0.42, 0.035, 0.73)[i]
        body.hue = body.base_hue
        body.wobble_frequency = rng.uniform(0.18, 0.43)
        body.wobble_phase = rng.uniform(0, 2*math.pi)
        body.direction = (1, -1, 1, -1)[i]
        body.angular_speed = body.direction*0.5
        body.pen_speed = (18.0, 20.0, 22.0, 19.0)[i]
        body.drawing_mode = "recomputed" if i < 2 and family == LEGACY else "ink"
        body.trail_offset = int(rng.integers(RECOMPUTED_POINTS))
        bodies.append(body)
    return bodies


def place_harmony_body(body, others, width=WORLD[0], height=WORLD[1]):
    """Enter in a free region so the fourth collider never pops into another."""
    low_x, high_x = MARGIN + body.radius, width - MARGIN - body.radius
    low_y, high_y = MARGIN + body.radius, height - MARGIN - body.radius
    best = None
    for x in np.linspace(low_x, high_x, 13):
        for y in np.linspace(low_y, high_y, 7):
            gap = min(math.hypot(x - other.x, y - other.y)
                      - body.radius - other.radius for other in others)
            if best is None or gap > best[0]:
                best = (gap, x, y)
    if best[0] < 0.05:
        return False
    _, body.x, body.y = best
    return True


class Physics:
    """Fixed circles, elastic equal-mass impulses, bounded adaptive substeps.

    Music changes body speed before this solver runs, never collider size.
    The solver projects penetrations before applying approaching-only impulses.
    """
    def __init__(self, width=16.0, height=9.0, margin=MARGIN):
        self.width, self.height, self.margin = width, height, margin
        self.contacts = 0

    def wall(self, b):
        for axis, size, velocity in (("x", self.width, "vx"), ("y", self.height, "vy")):
            lo, hi = self.margin + b.radius, size - self.margin - b.radius
            p, v = getattr(b, axis), getattr(b, velocity)
            if p < lo:
                setattr(b, axis, lo)
                setattr(b, velocity, abs(v))
            elif p > hi:
                setattr(b, axis, hi)
                setattr(b, velocity, -abs(v))

    def pair(self, a, b):
        dx, dy = b.x - a.x, b.y - a.y
        dist = math.hypot(dx, dy)
        limit = a.radius + b.radius
        if dist >= limit:
            return 0.0
        nx, ny = (dx / dist, dy / dist) if dist > 1e-12 else (1.0, 0.0)
        overlap = limit - dist
        push = (overlap + 1e-9) / 2
        a.x -= nx * push
        a.y -= ny * push
        b.x += nx * push
        b.y += ny * push
        relative = (b.vx - a.vx) * nx + (b.vy - a.vy) * ny
        if relative < 0:
            # Restitution=1: avoid cumulative loss of motion during long pieces.
            a.vx += relative * nx
            a.vy += relative * ny
            b.vx -= relative * nx
            b.vy -= relative * ny
            self.contacts += 1
        return overlap

    def step(self, bodies, dt):
        speed = max(math.hypot(b.vx, b.vy) for b in bodies)
        radius = min(b.radius for b in bodies)
        steps = max(1, math.ceil(speed * dt / (0.12 * radius)))
        for _ in range(steps):
            for b in bodies:
                b.x += b.vx * dt / steps
                b.y += b.vy * dt / steps
                self.wall(b)
            for _ in range(48):
                overlap = 0.0
                for i, a in enumerate(bodies):
                    for b in bodies[i + 1:]:
                        overlap = max(overlap, self.pair(a, b))
                for b in bodies:
                    self.wall(b)
                if overlap < 1e-8:
                    break


LIVE_RUN = 24


def polyline_runs(segments, colors, widths, run=LIVE_RUN):
    """Join consecutive pen segments into polylines of `run` segments each.

    A polyline is stroked as one path, so semi-transparent ink does not bead at
    every joint the way overlapping per-segment caps do, and seconds of held
    strokes stay cheap to draw. Each run takes its segments' mean color and
    width; a run spans a few hundredths of a second of pen travel, over which
    the fade changes opacity by about one percent.
    """
    starts = range(0, len(segments), run)
    paths = [np.vstack([segments[i:i+run, 0], segments[min(i+run, len(segments))-1, 1][None]])
             for i in starts]
    return (paths, np.array([colors[i:i+run].mean(axis=0) for i in starts]).reshape(-1, 4),
            np.array([widths[i:i+run].mean() for i in starts]))


class InkLayer:
    """The faint permanent trace of faded family strokes, in body-local pixels.

    Drawing minutes of ink as vectors every frame would be far too slow, so
    each settled chunk is drawn once into this raster ("over" compositing) in
    the trace style, and the raster is shown with the body's rotation, scale
    and position each frame. Its opacity is capped at TRACE_CAP, so the trace
    never fills into a solid shape. With TRACE_ALPHA = 0 nothing is drawn.
    """
    # Chunks (one video frame of pen travel each) drawn per canvas pass.
    BATCH = 2000

    def __init__(self, ax, half, pixels_per_unit, stroke_scale, plt, LineCollection):
        base = 2*half*pixels_per_unit
        supersample = 2 if base <= 1200 else 1
        size = max(8, math.ceil(base*supersample))
        # Canvas pixels per output pixel is the supersample factor.
        self.points_per_pixel = stroke_scale*supersample*72/100
        self.plt = plt
        self.fig = plt.figure(figsize=(size/100, size/100), dpi=100)
        self.fig.patch.set_alpha(0)
        local = self.fig.add_axes([0, 0, 1, 1])
        local.set_axis_off()
        local.patch.set_alpha(0)
        local.set(xlim=(-half, half), ylim=(-half, half))
        # Flat ends: neighbouring chunks abut instead of overlapping into beads.
        self.lines = LineCollection([], capstyle="butt", joinstyle="round")
        local.add_collection(self.lines)
        self.half = half
        # The canvas may round the requested inches by a pixel; match it exactly.
        width, height = self.fig.canvas.get_width_height()
        # Float accumulation keeps faint overlaid strokes from rounding away;
        # the displayed copy is 8-bit, which matplotlib resamples quickly.
        self.pixels = np.zeros((height, width, 4), dtype=np.float32)
        self.display = np.zeros((height, width, 4), dtype=np.uint8)
        self.image = ax.imshow(self.display, extent=(-half, half, -half, half),
                               interpolation="antialiased", zorder=0.8)

    def window(self, segments, points):
        """Canvas rows and columns that new strokes can touch."""
        height, width = self.pixels.shape[:2]
        columns = (segments[..., 0] + self.half)/(2*self.half)*width
        rows = (self.half - segments[..., 1])/(2*self.half)*height
        pad = float(points.max())*100/72/2 + 2
        return (slice(max(0, int(rows.min() - pad)), min(height, math.ceil(rows.max() + pad) + 1)),
                slice(max(0, int(columns.min() - pad)), min(width, math.ceil(columns.max() + pad) + 1)))

    def add(self, chunks):
        chunks = [c for c in chunks if len(c) > 1]
        if not chunks or TRACE_ALPHA <= 0:
            return
        # One polyline per chunk. Agg strokes a path once, so its joints never
        # double the alpha; per-segment strokes beaded visibly at 4K. A chunk
        # is one frame of pen travel, short enough for one color and width.
        paths = [c[:, 1:3] for c in chunks]
        colors = np.array([[*c[1:, 6:9].mean(axis=0), c[1:, 4].mean()*TRACE_ALPHA] for c in chunks])
        widths = np.array([c[1:, 3].mean()*TRACE_WIDTH for c in chunks])*self.points_per_pixel
        for start in range(0, len(paths), self.BATCH):
            part = slice(start, start + self.BATCH)
            self.lines.set_segments(paths[part])
            self.lines.set_color(np.clip(colors[part], 0, 1))
            self.lines.set_linewidth(widths[part])
            self.fig.canvas.draw()
            # Composite only where the new strokes are; the rest is untouched.
            rows, cols = self.window(np.concatenate(paths[part]), widths[part])
            new = np.asarray(self.fig.canvas.buffer_rgba())[rows, cols].astype(np.float32)/255
            # Agg's buffer holds straight (unpremultiplied) alpha.
            a = new[..., 3:]
            done = self.pixels[rows, cols]
            old = done[..., 3:].copy()
            alpha = a + old*(1 - a)
            done[..., :3] = (new[..., :3]*a + done[..., :3]*old*(1 - a))/np.maximum(alpha, 1e-6)
            done[..., 3:] = np.minimum(alpha, TRACE_CAP)
            self.display[rows, cols] = (done*255 + 0.5).astype(np.uint8)
        self.image.set_data(self.display)

    def place(self, ax, x, y, scale, angle, presence):
        from matplotlib.transforms import Affine2D
        self.image.set_transform(Affine2D().scale(scale).rotate(angle).translate(x, y) + ax.transData)
        self.image.set_visible(presence > 0)
        self.image.set_alpha(float(np.clip(presence, 0, 1)))

    def close(self):
        self.plt.close(self.fig)


class Renderer:
    """Reusable artists: clean render by default; explanatory labels optional."""
    def __init__(self, bodies, width=1280, height=720, guides=False, labels=False,
                 background=None, composition=None):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.patches import Circle
        self.plt = plt
        self.width, self.height = width, height
        self.stroke_scale = width / 1280
        self.composition = composition
        self.fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor="#080D16")
        self.ax = self.fig.add_axes([0, 0, 1, 1], facecolor="#080D16")
        self.ax.set(xlim=(0, 16), ylim=(0, 9), aspect="equal")
        self.ax.axis("off")
        if background is not None:
            self.ax.imshow(background, extent=(0, 16, 0, 9), origin="upper", zorder=0)
        self.guides, self.show_labels = guides, labels
        self.lines, self.circles, self.labels, self.tips = [], [], [], []
        self.glows = []
        # Accumulated family strokes; legacy bodies draw only vector ink.
        self.layers = {}
        for b in bodies:
            if b.family != LEGACY:
                zoom = composition[b.index][2] if composition is not None else 1.0
                self.layers[b.index] = InkLayer(self.ax, b.radius, width/16*zoom,
                                                self.stroke_scale, plt, LineCollection)
        self.ax.set(xlim=(0, 16), ylim=(0, 9))
        for b in bodies:
            # Flat ends avoid alpha accumulating into beads at every tiny
            # tessellation segment; the narrow opaque stroke masks the joins.
            glow = LineCollection([], capstyle="butt", zorder=1)
            self.ax.add_collection(glow)
            self.glows.append(glow)
            # Family heads are drawn as polyline runs (see polyline_runs).
            line = LineCollection([], capstyle="round" if b.family == LEGACY else "butt",
                                  joinstyle="round", zorder=2)
            self.ax.add_collection(line)
            self.lines.append(line)
            tip_scale = 1.0 if b.family == LEGACY else FAMILY_WIDTH
            self.tips.append(self.ax.plot([], [], marker="o", markersize=1.1*self.stroke_scale*tip_scale,
                                          color=COLORS[b.index], linestyle="none", alpha=0.95)[0])
            circle = Circle((b.x, b.y), b.radius, fill=False, color=COLORS[b.index],
                            linewidth=0.65, alpha=0.35, visible=guides)
            self.ax.add_patch(circle)
            self.circles.append(circle)
            self.labels.append(self.ax.text(b.x, b.y - b.radius + 0.05, LABELS[b.index],
                                           color=COLORS[b.index], fontsize=7, ha="center",
                                           va="bottom", visible=labels, alpha=0.8))

    def draw(self, bodies, frame_idx=0):
        for b, line, glow, circle, label, tip in zip(bodies, self.lines, self.glows, self.circles, self.labels, self.tips):
            layer = self.layers.get(b.index)
            if layer is not None:
                layer.add(b.drain_settled())
                x, y, zoom = (b.x, b.y, 1.0) if self.composition is None else self.composition[b.index]
                layer.place(self.ax, x, y, b.display_scale*zoom, b.angle, b.presence)
            circle.center = b.x, b.y
            label.set_position((b.x, b.y - b.radius + 0.05))
            circle.set_visible(self.guides and b.presence > 0)
            label.set_visible(self.show_labels and b.presence > 0)
            circle.set_alpha(0.35 * b.presence)
            label.set_alpha(0.8 * b.presence)
            if b.presence <= 0:
                line.set_segments([])
                glow.set_segments([])
                tip.set_data([], [])
                continue
            segments, colors, widths = b.strokes(frame_idx, detail=max(4, math.ceil(self.width/1000)))
            if self.composition is not None:
                x, y, scale = self.composition[b.index]
                segments = (segments - [b.x, b.y]) * scale + [x, y]
            colors = colors.copy()
            colors[:, 3] *= b.presence
            if len(colors):
                head = segments[-1, 1]
                tip.set_data([head[0]], [head[1]])
                tip.set_color(colors[-1])
            else:
                tip.set_data([], [])
            halo_segments, halo_colors, halo_widths = segments, colors, widths
            if b.family != LEGACY and len(segments):
                # Only the pen head glows; the held figure stays crisp.
                n = min(b.head_count(), len(segments))
                halo_segments, halo_colors, halo_widths = polyline_runs(
                    segments[len(segments)-n:], colors[len(colors)-n:], widths[len(widths)-n:])
                segments, colors, widths = polyline_runs(segments, colors, widths)
            line.set_segments(segments)
            line.set_color(colors)
            # Matplotlib uses points; convert the style's pixel width at 100 DPI.
            line.set_linewidth(widths * self.stroke_scale * 72/100)
            glow.set_segments(halo_segments)
            halo = halo_colors.copy()
            halo[:, 3] *= 0.025 + 0.045 * b.glow
            glow.set_color(halo)
            glow.set_linewidth(halo_widths * self.stroke_scale * (2.5 + 1.5*b.glow))
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3].copy()

    def save(self, path, dpi=300):
        # Pixel dimensions are established at render time; DPI is print metadata.
        from PIL import Image
        self.fig.canvas.draw()
        Image.fromarray(np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3]).save(path, dpi=(dpi, dpi))

    def close(self):
        for layer in self.layers.values():
            layer.close()
        self.plt.close(self.fig)
