"""
analyze_validation_results.py

Run this AFTER validate_squat_angles.py has produced
squat_angle_validation_results.csv.

This does two things:
  1. Flags any files where MediaPipe had low pose-detection coverage
     (frames_with_pose much lower than frames_total) - those files'
     angle numbers are less trustworthy and may be adding noise to
     the "overlap" you're seeing between correct/wrong categories.
  2. Tries a range of possible knee-angle thresholds and reports which
     one best separates "correct" from "wrong_*" in your actual data,
     so you pick a threshold backed by evidence instead of a guess.
"""

import csv

INPUT_CSV = "squat_angle_validation_results.csv"
COVERAGE_WARNING_THRESHOLD = 0.7  # flag files where <70% of frames had a detected pose


def load_results(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["frames_total"] = int(row["frames_total"])
            row["frames_with_pose"] = int(row["frames_with_pose"])
            row["min_knee_angle"] = float(row["min_knee_angle"])
            row["max_knee_angle"] = float(row["max_knee_angle"])
            row["avg_knee_angle"] = float(row["avg_knee_angle"])
            row["avg_latency_ms"] = float(row["avg_latency_ms"])
            rows.append(row)
    return rows


def check_coverage(rows):
    print("=" * 60)
    print("STEP 1: POSE DETECTION COVERAGE CHECK")
    print("=" * 60)
    flagged = []
    for r in rows:
        coverage = r["frames_with_pose"] / r["frames_total"] if r["frames_total"] else 0
        if coverage < COVERAGE_WARNING_THRESHOLD:
            flagged.append((r["file"], r["category"], coverage))

    if not flagged:
        print(f"All files have >= {COVERAGE_WARNING_THRESHOLD*100:.0f}% pose detection coverage. Good.")
    else:
        print(f"{len(flagged)} file(s) had LOW pose detection coverage "
              f"(< {COVERAGE_WARNING_THRESHOLD*100:.0f}% of frames):\n")
        for fname, cat, cov in flagged:
            print(f"   {fname}  [{cat}]  -> only {cov*100:.1f}% of frames had a detected pose")
        print("\nThese files' min/max angle numbers are less reliable and may be")
        print("contributing to the overlap you're seeing between categories.")
        print("Consider excluding them or re-checking the recording (lighting, framing, occlusion).")
    print()
    return flagged


def find_best_threshold(rows):
    print("=" * 60)
    print("STEP 2: THRESHOLD SEARCH")
    print("=" * 60)

    correct = [r["min_knee_angle"] for r in rows if r["category"] == "correct"]
    wrong = [r["min_knee_angle"] for r in rows if r["category"] != "correct"]

    if not correct or not wrong:
        print("Need at least one 'correct' and one 'wrong_*' file to compare. Skipping.")
        return

    print(f"correct: n={len(correct)}, min={min(correct):.1f}, max={max(correct):.1f}, "
          f"avg={sum(correct)/len(correct):.1f}")
    print(f"wrong:   n={len(wrong)}, min={min(wrong):.1f}, max={max(wrong):.1f}, "
          f"avg={sum(wrong)/len(wrong):.1f}")
    print()

    # Try every integer threshold in the relevant range and see which
    # gives the best split (correct squats go BELOW threshold, i.e. deeper bend)
    candidate_range = range(90, 181)
    best_threshold = None
    best_accuracy = -1

    for t in candidate_range:
        correct_pass = sum(1 for a in correct if a <= t)   # correct squats that dip below threshold = TP
        wrong_pass = sum(1 for a in wrong if a <= t)        # wrong squats incorrectly below threshold = FP
        correct_total = len(correct)
        wrong_total = len(wrong)

        tp = correct_pass
        fn = correct_total - correct_pass
        fp = wrong_pass
        tn = wrong_total - wrong_pass

        accuracy = (tp + tn) / (correct_total + wrong_total)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = t

    print(f"BEST THRESHOLD FOUND: knee_angle <= {best_threshold} degrees counts as a valid squat")
    print(f"This separates your labeled data with {best_accuracy*100:.1f}% accuracy.\n")

    if best_accuracy < 0.85:
        print("NOTE: accuracy is below 85%, meaning a single angle threshold alone")
        print("doesn't cleanly separate correct vs wrong in your data. This is expected")
        print("if 'wrong' refers to posture issues (not just shallow depth) - e.g.")
        print("wrong_front/wrong_side may be about alignment, not how low they went.")
        print("Next step would be adding a second check (e.g. torso lean angle, or")
        print("knee horizontal alignment) rather than relying on knee angle alone.")

    return best_threshold


def main():
    rows = load_results(INPUT_CSV)
    check_coverage(rows)
    find_best_threshold(rows)


if __name__ == "__main__":
    main()