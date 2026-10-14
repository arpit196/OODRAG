"""Inspect assessment text as it streams, before structured JSON parsing."""
import re

# Preserve ordinary tabs/newlines/CR and Unicode policy text (e.g. £ and bullets).
FORBIDDEN_CONTROLS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
REPEATED_CONTROL_ESCAPES = re.compile(
    r"(?:\\u(?:000[0-8bcef]|001[0-9a-f]|007f|008[0-9a-f]|009[0-9a-f])){8}",
    re.IGNORECASE,
)


class UnsafeAssessmentOutputError(ValueError):
    def __init__(self, pattern):
        super().__init__(f"Assessment stream rejected: {pattern}")
        self.guard_pattern = pattern
        self.raw_response = None
        self.stream_aborted = False


class AssessmentOutputGuard:
    def __init__(self):
        self.tail = ""

    def feed(self, text):
        if FORBIDDEN_CONTROLS.search(text):
            raise UnsafeAssessmentOutputError("forbidden_control_character")
        window = self.tail + text
        if REPEATED_CONTROL_ESCAPES.search(window):
            raise UnsafeAssessmentOutputError("repeated_control_character_escapes")
        # Keep enough overlap to detect escapes split across network deltas.
        self.tail = window[-256:]


def reject_decoded_controls(value):
    """Catch even a single escaped control after JSON decoding, in any field."""
    if isinstance(value, str):
        if FORBIDDEN_CONTROLS.search(value):
            raise UnsafeAssessmentOutputError("forbidden_control_character")
    elif isinstance(value, dict):
        for key, item in value.items():
            reject_decoded_controls(key)
            reject_decoded_controls(item)
    elif isinstance(value, list):
        for item in value:
            reject_decoded_controls(item)
