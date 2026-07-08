import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from detectors.sit_to_stand_detector_v2 import SitToStandRepCounter
from detectors.upper_trapezius_detector import UpperTrapRepCounter
from detectors.pendulum_detector import PendulumRepCounter
from detectors.shoulder_circle_detector import ShoulderCircleRepCounter

def test_sit_to_stand():
    counter = SitToStandRepCounter(leg_stand_threshold=160, leg_sit_threshold=110)
    state, rep, _ = counter.update(100, 150)
    assert state == 'sitting'
    assert rep == 0
    state, rep, _ = counter.update(165, 150)
    assert state == 'standing'
    assert rep == 0
    state, rep, just_completed = counter.update(105, 150)
    assert state == 'sitting'
    assert rep == 1
    assert just_completed == True

def test_upper_trap():
    counter = UpperTrapRepCounter(stretch_threshold=30.0, upright_threshold=15.0)
    state, rep, _ = counter.update(5.0, 'left')
    assert state == 'upright'
    assert rep == 0
    state, rep, _ = counter.update(35.0, 'left')
    assert state == 'stretching_left'
    assert rep == 0
    state, rep, just_completed = counter.update(10.0, 'left')
    assert state == 'upright'
    assert rep == 1
    assert just_completed == True

def test_pendulum():
    counter = PendulumRepCounter(forward_threshold=65, backward_threshold=115, neutral_min=75, neutral_max=105, max_torso_angle=120)
    state, rep, _ = counter.update(90, 150)
    assert state == 'invalid_posture'
    state, rep, _ = counter.update(90, 90)
    assert state == 'neutral'
    state, rep, _ = counter.update(50, 90)
    assert state == 'swinging_forward'
    state, rep, _ = counter.update(125, 90)
    assert state == 'swinging_backward'
    state, rep, just_completed = counter.update(90, 90)
    assert state == 'neutral'
    assert rep == 1
    assert just_completed == True

def test_shoulder_circle():
    counter = ShoulderCircleRepCounter()
    state, rep, _ = counter.update(45.0)
    assert state == 'QUADRANT_1'
    state, rep, _ = counter.update(135.0)
    assert state == 'QUADRANT_2'
    state, rep, _ = counter.update(225.0)
    assert state == 'QUADRANT_3'
    state, rep, just_completed = counter.update(315.0)
    assert state == 'QUADRANT_4'
    assert rep == 1
    assert just_completed == True
    state, rep, _ = counter.update(45.0)
    assert rep == 1
