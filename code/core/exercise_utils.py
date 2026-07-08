import json
from detectors.squat_angle_detector import SquatExercise
from detectors.pendulum_detector import PendulumExercise
from detectors.shoulder_circle_detector import ShoulderCircleExercise
from detectors.sit_to_stand_detector_v2 import SitToStandExercise
from detectors.upper_trapezius_detector import UpperTrapeziusExercise

EXERCISE_REGISTRY = {
    'mini_squat': SquatExercise,
    'full_squat': SquatExercise,
    'pendulum': PendulumExercise,
    'shoulder_circle': ShoulderCircleExercise,
    'sit_to_stand': SitToStandExercise,
    'upper_trapezius': UpperTrapeziusExercise
}

def load_config():
    with open('exercises_config.json', 'r') as f:
        return json.load(f)

def select_exercise(config_data):
    print('\n' + '=' * 30)
    print('  SMART MIRROR EXERCISE TRACKER')
    print('=' * 30)
    keys = [k for k in config_data.keys() if k != 'front_squat']
    for i, key in enumerate(keys):
        print(f"[{i + 1}] {config_data[key]['name']}")
    while True:
        try:
            choice = int(input(f'\nSelect an exercise [1-{len(keys)}]: '))
            if 1 <= choice <= len(keys):
                selected_key = keys[choice - 1]
                return (selected_key, config_data[selected_key])
            else:
                print('Invalid choice, try again.')
        except ValueError:
            print('Please enter a valid number.')

def create_exercise(key, intrinsics, config):
    if key not in EXERCISE_REGISTRY:
        raise ValueError(f'Unknown exercise {key}')
    exercise_class = EXERCISE_REGISTRY[key]
    return exercise_class(config, intrinsics)

class SessionTracker:
    def __init__(self):
        self.correct_reps = 0
        self.incorrect_reps = 0
        self.mistake_counts = {}
        self.in_error_state = False

    def update(self, state_text, cue_text, just_completed):
        if just_completed:
            self.correct_reps += 1
        if state_text and state_text.lower().startswith('invalid'):
            if not self.in_error_state:
                self.incorrect_reps += 1
                self.in_error_state = True
                mistake_reason = cue_text.split('-')[0].strip()
                self.mistake_counts[mistake_reason] = self.mistake_counts.get(mistake_reason, 0) + 1
        else:
            self.in_error_state = False

    def print_summary(self):
        print('\n' + '=' * 40)
        print('         WORKOUT SUMMARY')
        print('=' * 40)
        total_attempts = self.correct_reps + self.incorrect_reps
        accuracy = self.correct_reps / total_attempts * 100 if total_attempts > 0 else 0.0
        print(f'Total Reps Completed: {self.correct_reps}')
        print(f'Total Mistakes Made:  {self.incorrect_reps}')
        print(f'Overall Accuracy:     {accuracy:.1f}%\n')
        if self.incorrect_reps > 0:
            print('Areas for Improvement:')
            for reason, count in sorted(self.mistake_counts.items(), key=lambda x: x[1], reverse=True):
                print(f'  - {reason} ({count} times)')
        elif self.correct_reps > 0:
            print('Perfect form! Keep up the great work!')
        else:
            print('No reps attempted.')
        print('=' * 40 + '\n')
