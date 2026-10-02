import sys
import os
from types import SimpleNamespace
import numpy as np
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.pose_geometry_base import PoseGeometryBase, EMAFilter, mp_pose
from core.exercise_utils import load_config, EXERCISE_REGISTRY

INTRINSICS = SimpleNamespace(fx=600.0, fy=600.0, ppx=320.0, ppy=240.0)


class FakeDepthFrame:
    def get_units(self):
        return 0.001  # RealSense default: 1 unit = 1 mm


@pytest.fixture(scope='module')
def geom():
    g = PoseGeometryBase(depth_intrinsics=INTRINSICS)
    yield g
    g.close()


def test_depth_ignores_holes_and_noise(geom):
    depth = np.full((480, 640), 2000, dtype=np.uint16)
    depth[240, 320] = 0          # hole exactly at the joint
    depth[241, 321] = 9000       # one noisy outlier next to it
    p = geom._pixel_depth_to_3d(0.5, 0.5, FakeDepthFrame(), depth)
    assert p is not None
    assert p[2] == pytest.approx(2000.0)


def test_depth_all_holes_returns_none(geom):
    depth = np.zeros((480, 640), dtype=np.uint16)
    assert geom._pixel_depth_to_3d(0.5, 0.5, FakeDepthFrame(), depth) is None


def _landmarks(hip_xy, shoulder_xy):
    lms = [SimpleNamespace(x=0.0, y=0.0, z=0.0, visibility=0.0) for _ in range(33)]
    lms[mp_pose.PoseLandmark.LEFT_HIP.value] = SimpleNamespace(x=hip_xy[0], y=hip_xy[1], z=0.0, visibility=1.0)
    lms[mp_pose.PoseLandmark.LEFT_SHOULDER.value] = SimpleNamespace(x=shoulder_xy[0], y=shoulder_xy[1], z=0.0, visibility=1.0)
    return SimpleNamespace(landmark=lms)


def test_torso_upright_is_zero(geom):
    depth = np.full((480, 640), 2000, dtype=np.uint16)
    lean = geom.get_torso_lean_angle(_landmarks((0.5, 0.7), (0.5, 0.3)), 'left', FakeDepthFrame(), depth)
    assert lean == pytest.approx(0.0, abs=1.0)


def test_torso_lean_uses_pixel_aspect_without_depth(geom):
    # 100 px right and 100 px up on a 640x480 image is a 45 degree lean;
    # in raw normalised coordinates it would wrongly come out near 37 degrees.
    hip = (320 / 640, 300 / 480)
    shoulder = (420 / 640, 200 / 480)
    lean = geom.get_torso_lean_angle(_landmarks(hip, shoulder), 'left')
    assert lean == pytest.approx(45.0, abs=0.5)


def test_ema_keeps_last_value_on_missing_input():
    f = EMAFilter(alpha=0.5)
    assert f.update(10.0) == 10.0
    assert f.update(None) == 10.0
    assert f.update(20.0) == 15.0


def test_config_loads_from_any_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_config()
    for key in EXERCISE_REGISTRY:
        assert key in config
