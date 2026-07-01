import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from squat_angle_detector import SquatRepCounter

def test_perfect_rep():
    counter = SquatRepCounter(stand_threshold=160, squat_threshold=130, angle_min=60)
    
    # Start standing
    state, rep, _ = counter.update(170)
    assert state == "standing"
    assert rep == 0
    
    # Go down
    state, rep, _ = counter.update(120)
    assert state == "squatting"
    
    # Bottom out (good depth)
    state, rep, _ = counter.update(80)
    assert state == "squatting"
    
    # Stand back up
    state, rep, just_completed = counter.update(165)
    assert state == "standing"
    assert rep == 1
    assert just_completed == True

def test_too_deep_rep():
    counter = SquatRepCounter(stand_threshold=160, squat_threshold=130, angle_min=60)
    
    counter.update(170)
    counter.update(120) # enters squatting state
    
    state, rep, _ = counter.update(50)
    assert state == "invalid_depth"
    
    state, rep, just_completed = counter.update(165)
    assert state == "standing"
    assert rep == 0
    assert just_completed == False

def test_asymmetric_rep():
    counter = SquatRepCounter(stand_threshold=160, squat_threshold=130, max_angle_diff=20)
    
    counter.update(170, angle_diff=5)
    counter.update(120, angle_diff=5) # enters squatting
    
    state, rep, _ = counter.update(110, angle_diff=25)
    assert state == "invalid_asymmetric"
    
    state, rep, _ = counter.update(165, angle_diff=5)
    assert rep == 0

def test_wide_knees_rep():
    counter = SquatRepCounter(stand_threshold=160, squat_threshold=130, max_knee_dist_mm=500)
    
    counter.update(170, knee_dist=300)
    counter.update(120, knee_dist=300) # enters squatting
    
    state, rep, _ = counter.update(110, knee_dist=550)
    assert state == "invalid_wide_knees"
    
    state, rep, _ = counter.update(165, knee_dist=300)
    assert rep == 0

def test_bad_torso_lean_rep():
    counter = SquatRepCounter(stand_threshold=160, squat_threshold=130, max_torso_lean=55)
    
    counter.update(170, torso_lean=10)
    counter.update(120, torso_lean=10) # enters squatting
    
    state, rep, _ = counter.update(110, torso_lean=60)
    assert state == "invalid_back_posture"
    
    state, rep, _ = counter.update(165, torso_lean=10)
    assert rep == 0
