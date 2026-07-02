import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sit_to_stand_detector_v2 import SitToStandRepCounter
from upper_trapezius_detector import UpperTrapRepCounter
from pendulum_detector import PendulumRepCounter
from shoulder_circle_detector import ShoulderCircleRepCounter

def test_sit_to_stand():
    counter = SitToStandRepCounter(leg_stand_threshold=160, leg_sit_threshold=110)
    
    # Start sitting
    state, rep, _ = counter.update(100, 150)
    assert state == "sitting"
    assert rep == 0
    
    # Stand up
    state, rep, _ = counter.update(165, 150)
    assert state == "standing"
    assert rep == 0
    
    # Sit back down
    state, rep, just_completed = counter.update(105, 150)
    assert state == "sitting"
    assert rep == 1
    assert just_completed == True


def test_upper_trap():
    counter = UpperTrapRepCounter(stretch_threshold=30.0, upright_threshold=15.0)
    
    # Start upright
    state, rep, _ = counter.update(5.0, "left")
    assert state == "upright"
    assert rep == 0
    
    # Stretch left
    state, rep, _ = counter.update(35.0, "left")
    assert state == "stretching_left"
    assert rep == 0
    
    # Return upright
    state, rep, just_completed = counter.update(10.0, "left")
    assert state == "upright"
    assert rep == 1
    assert just_completed == True


def test_pendulum():
    counter = PendulumRepCounter(forward_threshold=65, backward_threshold=115, neutral_min=75, neutral_max=105, max_torso_angle=120)
    
    # Invalid posture
    state, rep, _ = counter.update(90, 150) # standing too straight
    assert state == "invalid_posture"
    
    # Valid posture (bent over, arm neutral)
    state, rep, _ = counter.update(90, 90)
    assert state == "neutral"
    
    # Swing forward (angle decreases)
    state, rep, _ = counter.update(50, 90)
    assert state == "swinging_forward"
    
    # Swing backward (angle increases)
    state, rep, _ = counter.update(125, 90)
    assert state == "swinging_backward"
    
    # Return neutral
    state, rep, just_completed = counter.update(90, 90)
    assert state == "neutral"
    assert rep == 1
    assert just_completed == True


def test_shoulder_circle():
    counter = ShoulderCircleRepCounter()
    
    # Start Q1
    state, rep, _ = counter.update(45.0)
    assert state == "QUADRANT_1"
    
    # Move Q2
    state, rep, _ = counter.update(135.0)
    assert state == "QUADRANT_2"
    
    # Move Q3
    state, rep, _ = counter.update(225.0)
    assert state == "QUADRANT_3"
    
    # Move Q4
    state, rep, just_completed = counter.update(315.0)
    assert state == "QUADRANT_4"
    assert rep == 1
    assert just_completed == True
    
    # Keep rotating to Q1 (starts next rep)
    state, rep, _ = counter.update(45.0)
    assert rep == 1
