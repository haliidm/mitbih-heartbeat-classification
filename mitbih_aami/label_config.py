"""MIT-BIH sampling, labels, and random-window split configuration."""

FS = 360
WINDOW_SECONDS = 16
WINDOW_SAMPLES = WINDOW_SECONDS * FS
WINDOW_HALF = WINDOW_SAMPLES // 2
PAD_SAMPLES = WINDOW_SAMPLES
INFERENCE_STRIDE = 8 * FS
LABEL_RADIUS_MS = 180
MAX_LABEL_RADIUS = round(LABEL_RADIUS_MS * FS / 1000)
PEAK_REFINEMENT_RADIUS_MS = 50
PEAK_REFINEMENT_RADIUS = round(PEAK_REFINEMENT_RADIUS_MS * FS / 1000)

CLASS_NAMES = ("background", "N", "S", "V", "F", "Q")
NUM_CLASSES = len(CLASS_NAMES)

SYMBOL_TO_CLASS = {
    "N": 1, "L": 1, "R": 1, "e": 1, "j": 1,
    "A": 2, "a": 2, "J": 2, "S": 2,
    "V": 3, "E": 3,
    "F": 4,
    "/": 5, "f": 5, "Q": 5,
}

CENTER_CLASSES = (2, 3, 4, 5)
SPLIT_RATIOS = (0.70, 0.10, 0.20)


