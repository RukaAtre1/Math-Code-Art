"""Regression checks for independent V3: rotation, music mapping and bounds."""
import math
import unittest
from unittest import mock

import numpy as np

from music_spirograph.art_direction import TRACK_CONFIG
import music_spirograph.exhibition_v3 as ev
from music_spirograph.curve_families import (FADE_FIGURES, FAMILIES, FAMILY_SPIN, FIGURE_SECONDS, HOLD_FIGURES, LEGACY,
    MORPH_SECONDS, PROTOTYPES, SWITCH_DWELL_SECONDS, TRACE_CAP, evaluate, recommend_family)
from music_spirograph.exhibition_v3 import (make_bodies, Physics, PADDING, MAX_INK_CHORD, Renderer,
    TRAIL_SECONDS, RECOMPUTED_POINTS, RECOMPUTED_TRAIL_FRACTION,
    harmony_activity_score, sustained_harmony, place_harmony_body, TrackProfile)


def features(level=0.3):
    peak=0.20+0.70*level
    return dict(onset=level, energy=level, timbre=level, beat=level, bass=level,
                bandwidth=level, flux=level, tempo=1.0, activity=1.0,
                chroma=np.array([peak]+[(1-peak)/11]*11))


class V3Tests(unittest.TestCase):
    def test_temporary_harmony_requires_sustained_polyphonic_activity(self):
        solo = np.zeros((12, 1))
        solo[0, 0] = 0.8
        solo[7, 0] = 0.1
        solo[4, 0] = 0.1
        chord = np.zeros((12, 1))
        chord[[0, 4, 7], 0] = [0.38, 0.34, 0.28]
        self.assertLess(float(harmony_activity_score(solo, np.ones(1))[0]), 0.1)
        self.assertGreater(float(harmony_activity_score(chord, np.ones(1))[0]), 0.38)
        signal = np.r_[np.zeros(5), np.ones(3), np.zeros(5),
                       np.ones(8), np.zeros(5), np.ones(2), np.zeros(12)]
        active = sustained_harmony(signal, 0.1)
        self.assertFalse(active[:13].any())
        self.assertTrue(active[20])
        self.assertTrue(active[25])
        self.assertFalse(active[-1])

    def test_fourth_body_enters_clear_of_other_colliders(self):
        bodies = make_bodies(4)
        self.assertTrue(place_harmony_body(bodies[3], bodies[:3]))
        self.assertTrue(all(math.hypot(bodies[3].x-b.x, bodies[3].y-b.y)
                            >= bodies[3].radius+b.radius+0.05-1e-9 for b in bodies[:3]))
        bodies[3].presence = 0
        bodies[3].update_presence(True, 0.225)
        self.assertAlmostEqual(bodies[3].presence, 0.5)
        bodies[3].update_presence(False, 0.8)
        self.assertEqual(bodies[3].presence, 0)

    def test_old_and_new_ink_rotate_as_one_body(self):
        b=make_bodies()[2]
        for _ in range(30):
            b.update_visual(features(), 1/30)
        original=list(b.ink)
        points=np.asarray(original)[:, 1:3]
        b.angle=0.0
        zero=b.world_points(points)-[b.x,b.y]
        b.angle=math.pi/2
        rotated=b.world_points(points)-[b.x,b.y]
        np.testing.assert_allclose(rotated, np.column_stack([-zero[:,1],zero[:,0]]), atol=1e-12)
        self.assertEqual(list(b.ink), original)
        self.assertGreater(np.linalg.norm(rotated[0]-zero[0]),0.1)

    def test_every_role_rotates_and_changes_multiple_geometry_parameters(self):
        for a,b in zip(make_bodies(4),make_bodies(4)):
            for _ in range(60):
                a.update_visual(features(0.05),1/30)
                b.update_visual(features(0.95),1/30)
            self.assertGreater(abs(a.angle),0.1)
            changed=np.abs(a.geometry()[:4]-b.geometry()[:4])>1e-3
            self.assertGreaterEqual(int(changed.sum()),2)
            self.assertNotAlmostEqual(a.angular_speed,b.angular_speed)

    def test_seed_changes_geometry_but_is_reproducible(self):
        a,b,c=make_bodies(seed=17),make_bodies(seed=17),make_bodies(seed=29)
        for x,y,z in zip(a,b,c):
            np.testing.assert_equal(x.geometry(),y.geometry())
            self.assertFalse(np.allclose(x.geometry(),z.geometry()))

    def test_audio_only_changes_new_local_ink(self):
        b=make_bodies()[2]
        b.update_visual(features(0.1),1/30)
        first=list(b.ink)
        b.update_visual(features(0.9),1/30)
        self.assertEqual(list(b.ink)[:len(first)],first)
        self.assertGreater(len(b.ink),len(first))

    def test_default_three_mix_two_recomputed_curves_with_one_ink_body(self):
        bodies = make_bodies()
        self.assertEqual([b.drawing_mode for b in bodies],
                         ["recomputed", "recomputed", "ink"])
        body = bodies[0]
        body.update_visual(features(0.1), 1/30)
        body.angle = 0.0
        old, _, _ = body.strokes(0)
        body.update_visual(features(0.9), 1/30)
        body.angle = 0.0
        new, _, _ = body.strokes(0)
        self.assertEqual(len(new), int(RECOMPUTED_POINTS*RECOMPUTED_TRAIL_FRACTION)-1)
        self.assertEqual(len(body.ink), 0)
        self.assertGreater(np.linalg.norm(new-old), 0.1)
        shifted, _, _ = body.strokes(1)
        self.assertGreater(np.linalg.norm(shifted-new), 0.1)

    def test_music_changes_body_speed_without_changing_collider(self):
        quiet, loud = make_bodies(4)[0], make_bodies(4)[0]
        for _ in range(30):
            quiet.update_motion(features(0.0), 1/30)
            loud.update_motion(features(1.0), 1/30)
        self.assertGreater(math.hypot(loud.vx, loud.vy), math.hypot(quiet.vx, quiet.vy)+0.8)
        self.assertLessEqual(math.hypot(loud.vx, loud.vy), 2.23)
        self.assertEqual(quiet.radius, loud.radius)

    def test_ink_history_has_short_fade(self):
        body = make_bodies(4)[2]
        self.assertEqual(body.trail_seconds, TRAIL_SECONDS)
        for _ in range(120):
            body.update_visual(features(0.5), 1/30)
        self.assertGreaterEqual(body.ink[0][0], body.elapsed-TRAIL_SECONDS)
        self.assertGreater(len(body.ink), 0)

    def test_whole_track_profile_changes_starting_geometry(self):
        gentle = TrackProfile(0.2, 0.1, 0.2, 0.1, 0.2, 0.1)
        dramatic = TrackProfile(0.9, 0.8, 0.7, 0.8, 0.7, 0.7)
        a = make_bodies(4, seed=17, track_profile=gentle)
        b = make_bodies(4, seed=17, track_profile=dramatic)
        for left, right in zip(a, b):
            self.assertFalse(np.allclose(left.geometry()[:4], right.geometry()[:4]))

    def test_rotated_strokes_stay_in_colliders(self):
        bodies=make_bodies(4)
        physics=Physics()
        for frame in range(600):
            f=features(0.5+0.5*math.sin(frame/13))
            for body in bodies:
                body.update_visual(f,1/30)
                body.update_motion(f,1/30)
                if frame%30==0:
                    if body.drawing_mode == "recomputed":
                        segments, _, _ = body.strokes(frame)
                        world = segments.reshape(-1, 2)-[body.x,body.y]
                    else:
                        x,y=body.local_points()
                        world=body.world_points(np.column_stack([x,y]))-[body.x,body.y]
                    self.assertLessEqual(float(np.linalg.norm(world,axis=1).max()),body.radius-PADDING+1e-9)
            physics.step(bodies,1/30)
            for i,a in enumerate(bodies):
                for b in bodies[i+1:]:
                    self.assertGreaterEqual(math.hypot(a.x-b.x,a.y-b.y),a.radius+b.radius-1e-7)


def all_ink(body):
    """Every family ink sample in drawing order: settled chunks, then live ink."""
    parts = []
    for chunk in body.settled:
        # Each later chunk repeats the previous chunk's last sample.
        parts.append(chunk if not parts else chunk[1:])
    if body.ink:
        parts.append(np.asarray(body.ink))
    return np.concatenate(parts) if parts else np.empty((0, 9))


def stroke_points(body, frame):
    if body.drawing_mode == "recomputed":
        segments, _, _ = body.strokes(frame, detail=4)
        return segments.reshape(-1, 2)-[body.x, body.y]
    if body.family != LEGACY:
        return body.world_points(all_ink(body)[:, 1:3])-[body.x, body.y]
    x, y = body.local_points()
    return body.world_points(np.column_stack([x, y]))-[body.x, body.y]


def sectors(points):
    """Distinct 10-degree sectors around the center that hold ink."""
    angles = np.angle(points[:, 0]+1j*points[:, 1])
    return len(np.unique(np.floor((angles + math.pi)/(math.pi/18))))


class CurveFamilyTests(unittest.TestCase):
    def test_default_bodies_keep_the_original_double_rotor(self):
        self.assertTrue(all(b.family == LEGACY for b in make_bodies(4)))
        with self.assertRaises(ValueError):
            make_bodies(3, family="spiral")

    def test_each_exhibition_piece_has_its_own_family(self):
        families = [c.family for c in TRACK_CONFIG.values()]
        self.assertEqual(len(set(families)), 5)
        self.assertTrue(set(families) <= set(FAMILIES))

    def test_every_figure_closes_after_one_cycle(self):
        # Accumulated passes land on the same lines instead of drifting into
        # a filled disc: the pen returns to its start after one cycle.
        for name, family in FAMILIES.items():
            for motifs in family.motifs:
                for motif in motifs:
                    for pen in (0.4, 1.0, 2.5):
                        ends, _ = evaluate(name, np.array([0.3, 0.3 + 2*math.pi]), motif, 1.1, 0.9, 1.2, pen)
                        self.assertAlmostEqual(abs(ends[0]-ends[1]), 0.0, places=9)

    def test_each_figure_is_one_curve_with_its_symmetry(self):
        s = np.linspace(0, 2*math.pi, 500, endpoint=False)
        for name, family in FAMILIES.items():
            for motifs in family.motifs:
                for motif in motifs:
                    # No shorter period: one cycle draws the figure once, not twice.
                    self.assertEqual(math.gcd(*motif), 1, (name, motif))
                    m = family.symmetry(motif)
                    self.assertGreaterEqual(m, 3, (name, motif))
                    first, _ = evaluate(name, s, motif, 1.1, 0.9, 1.2, 1.8)
                    turned, _ = evaluate(name, s + 2*math.pi/m, motif, 1.1, 0.9, 1.2, 1.8)
                    np.testing.assert_allclose(turned, first*np.exp(2j*math.pi*motif[0]/m), atol=1e-9)

    def test_bound_caps_every_point_of_every_motif(self):
        s = np.linspace(0, 6*math.pi, 4000)
        for name, family in FAMILIES.items():
            for motifs in family.motifs:
                for motif in motifs:
                    for controls in ((0.7, 0.6, 0.6, 0.4), (1.3, 1.4, 1.4, 3.0)):
                        z, bound = evaluate(name, s, motif, *controls)
                        self.assertTrue(np.all(np.abs(z) <= bound+1e-12))

    def test_every_family_body_draws_with_its_pen(self):
        for name in FAMILIES:
            for body in make_bodies(4, family=name):
                self.assertEqual(body.drawing_mode, "ink")

    def test_family_pen_path_is_recorded_without_gaps(self):
        body = make_bodies(3, family="rotor_crystal")[0]
        frames = int(3*FIGURE_SECONDS*30)
        for frame in range(frames):
            body.update_visual(features(0.5+0.4*math.sin(frame/11)), 1/30)
        ink = all_ink(body)
        times = ink[:, 0]
        # One unbroken pen path from the first frame to now: no sample dropped.
        self.assertLessEqual(times[0], 1/30)
        self.assertAlmostEqual(times[-1], body.elapsed, places=9)
        self.assertTrue(np.all(np.diff(times) > 0))
        self.assertLessEqual(float(np.diff(times).max()), 1/30+1e-9)
        # Only the figure just drawn and the fading stretch are still vectors.
        self.assertLessEqual(body.cycle - body.ink[0][9], 2*math.pi*(HOLD_FIGURES + FADE_FIGURES))
        self.assertGreater(body.cycle - body.ink[0][9], 2*math.pi*HOLD_FIGURES)
        self.assertEqual(sectors(ink[:, 1:3]), 36)

    def test_hidden_fourth_voice_keeps_its_strokes(self):
        body = make_bodies(4, family="rotor_waltz")[3]
        for _ in range(90):
            body.update_visual(features(0.6), 1/30)
        before = len(all_ink(body))
        while body.presence > 0:
            body.update_presence(False, 1/30)
        self.assertEqual(len(body.ink), 0)
        self.assertEqual(len(all_ink(body)), before)

    @mock.patch.object(ev, "TRACE_ALPHA", 0.15)
    def test_trace_raster_only_grows_and_follows_the_body(self):
        bodies = make_bodies(3, family="rotor_corona")
        renderer = Renderer(bodies, 320, 180)
        try:
            coverage = []
            for frame in range(int(4*FIGURE_SECONDS*30)):
                for body in bodies:
                    body.update_visual(features(0.5+0.4*math.sin(frame/9)), 1/30)
                if frame % 15 == 0:
                    renderer.draw(bodies, frame)
                    coverage.append(float(renderer.layers[0].pixels[..., 3].sum()))
            self.assertGreater(coverage[-1], 0)
            self.assertTrue(all(b >= a - 1e-3 for a, b in zip(coverage, coverage[1:])))
            renderer.draw(bodies)
            self.assertTrue(all(not body.settled for body in bodies))
            image = renderer.layers[0].image
            corner = image.get_transform().transform((0.0, 0.0))
            expected = renderer.ax.transData.transform((bodies[0].x, bodies[0].y))
            np.testing.assert_allclose(corner, expected, atol=1e-6)
        finally:
            renderer.close()

    @mock.patch.object(ev, "TRACE_ALPHA", 0.15)
    def test_raster_layer_matches_any_canvas_size(self):
        # Requested inches can round to a different pixel count than asked.
        bodies = make_bodies(3, family="rotor_crystal")
        for width in (320, 1366, 1918):
            renderer = Renderer(bodies, width, width*9//16, composition=((4, 3, 2.37), (8, 5, 0.71), (12, 4, 1.13)))
            try:
                chunk = np.zeros((3, 9))
                chunk[:, 0] = (0.0, 0.1, 0.2)
                chunk[:, 1:3] = ((0, 0), (0.5, 0.2), (0.6, -0.4))
                chunk[:, 3:5] = (1.6, 0.9)
                chunk[:, 6:9] = (0.8, 0.5, 0.9)
                for layer in renderer.layers.values():
                    layer.add([chunk])
                    self.assertGreater(float(layer.pixels[..., 3].max()), 0)
            finally:
                renderer.close()

    def test_last_figure_is_held_then_fades_to_the_trace(self):
        body = make_bodies(3, family="rotor_crystal")[0]
        for frame in range(int(3*FIGURE_SECONDS*30)):
            body.update_visual(features(0.5+0.4*math.sin(frame/7)), 1/30)
        _, colors, _ = body.strokes()
        live = list(body.ink)
        if body.settled_tail is not None:
            live.insert(0, body.settled_tail)
        history = np.asarray(live)[1:]
        figures = (body.cycle - history[:, 9])/(2*math.pi)
        ratio = colors[:, 3]/history[:, 4]
        held = figures <= HOLD_FIGURES
        # The whole last figure is at full strength, then strokes fade with age.
        self.assertTrue(held.any() and not held.all())
        np.testing.assert_allclose(ratio[held], 1.0)
        self.assertTrue(np.all(np.diff(ratio) >= -1e-9))
        self.assertLess(ratio[0], 0.05)

    def test_trace_never_exceeds_its_cap(self):
        bodies = make_bodies(3, family="rotor_crystal")
        renderer = Renderer(bodies, 320, 180)
        try:
            chunk = np.zeros((40, 9))
            chunk[:, 0] = np.linspace(0, 1, 40)
            chunk[:, 1] = np.linspace(-0.8, 0.8, 40)
            chunk[:, 3:5] = (3.0, 1.0)
            chunk[:, 6:9] = (0.8, 0.5, 0.9)
            layer = renderer.layers[0]
            with mock.patch.object(ev, "TRACE_ALPHA", 0.0):
                layer.add([chunk])
                self.assertEqual(float(layer.pixels[..., 3].max()), 0.0)
            with mock.patch.object(ev, "TRACE_ALPHA", 1.0):
                for _ in range(20):
                    layer.add([chunk])
            self.assertLessEqual(float(layer.pixels[..., 3].max()), TRACE_CAP + 1e-6)
            self.assertGreater(float(layer.pixels[..., 3].max()), 0.0)
        finally:
            renderer.close()

    def test_family_strokes_stay_in_colliders_through_a_motif_switch(self):
        for name in FAMILIES:
            bodies = make_bodies(4, family=name)
            physics = Physics()
            # 15 fps keeps this long run quick; the bounds do not depend on it.
            for frame in range(int((SWITCH_DWELL_SECONDS+MORPH_SECONDS+2)*15)):
                f = features(0.5+0.5*math.sin(frame/3.5))
                f["phrase_change"] = 1.0
                for body in bodies:
                    body.update_visual(f, 1/15)
                    body.update_motion(f, 1/15)
                    if frame % 30 == 0:
                        reach = float(np.linalg.norm(stroke_points(body, frame), axis=1).max())
                        self.assertLessEqual(reach, body.radius-PADDING+1e-9, name)
                physics.step(bodies, 1/15)
            self.assertTrue(all(b.switches == 1 for b in bodies), name)
            self.assertTrue(all(b.motif_index == 1 or b.motif_target == 1 for b in bodies), name)

    def test_motif_switch_waits_for_dwell_and_cross_fades(self):
        body = make_bodies(3, family="rotor_crystal")[2]
        f = features(0.5)
        f["phrase_change"] = 1.0
        frames = 0
        while body.switches == 0:
            body.update_visual(f, 1/30)
            frames += 1
        self.assertGreaterEqual(frames/30, SWITCH_DWELL_SECONDS)
        self.assertEqual((body.motif_index, body.motif_target), (0, 1))
        for _ in range(int(MORPH_SECONDS*30)+4):
            body.update_visual(f, 1/30)
        self.assertEqual((body.motif_index, body.motif_target), (1, None))
        self.assertEqual(body.switches, 1)

    def test_family_ink_never_jumps_when_music_jumps(self):
        for name in FAMILIES:
            body = make_bodies(3, family=name)[2]
            for frame in range(90):
                f = features(0.05 if frame % 2 else 0.95)
                f["phrase_change"] = 1.0
                body.update_visual(f, 1/30)
            steps = np.linalg.norm(np.diff(all_ink(body)[:, 1:3], axis=0), axis=1)
            # A teleport would cross a large part of the collider; resampling
            # also keeps ordinary chords short enough to look curved.
            self.assertLess(float(steps.max()), 2*MAX_INK_CHORD, name)

    def test_drawing_speed_follows_the_music(self):
        quiet, loud = (make_bodies(3, family="rotor_corona")[0] for _ in range(2))
        for _ in range(60):
            quiet.update_visual(features(0.0), 1/30)
            loud.update_visual(features(1.0), 1/30)
        self.assertGreater(loud.cycle, quiet.cycle*1.5)

    def test_family_figures_turn_faster_than_the_double_rotor(self):
        legacy = make_bodies(3)[2]
        family = make_bodies(3, family="rotor_crystal")[2]
        for _ in range(60):
            legacy.update_visual(features(0.4), 1/30)
            family.update_visual(features(0.4), 1/30)
        self.assertAlmostEqual(family.angular_speed, legacy.angular_speed*FAMILY_SPIN, places=6)

    def test_auto_family_follows_the_most_similar_exhibition_piece(self):
        for name, values in PROTOTYPES.items():
            self.assertEqual(recommend_family(TrackProfile(*values)), name)


if __name__=='__main__':
    unittest.main()
