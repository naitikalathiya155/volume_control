"""
Gesture-Controlled Laptop Volume
--------------------------------
Pinch your thumb and index finger together/apart in front of the webcam
to lower/raise your Windows system volume in real time.

Pipeline:
  Webcam frame -> MediaPipe Hands (21 landmarks) -> pinch distance
  -> normalized 0..1 -> mapped to system volume (via pycaw) -> applied

Run on Windows only (pycaw is a Windows Core Audio wrapper).
"""

import cv2
import mediapipe as mp
import numpy as np
import math

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# ---------------------------------------------------------------
# Windows audio setup
# ---------------------------------------------------------------
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume_ctrl = cast(interface, POINTER(IAudioEndpointVolume))

vol_range = volume_ctrl.GetVolumeRange()   # (min_dB, max_dB, step)
MIN_VOL_DB, MAX_VOL_DB = vol_range[0], vol_range[1]

# ---------------------------------------------------------------
# MediaPipe Hands setup
# ---------------------------------------------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

# Smoothing so volume doesn't jump around with tiny hand jitter
smoothed_vol_percent = volume_ctrl.GetMasterVolumeLevelScalar() * 100


def calc_pinch_distance(landmarks, w, h):
    """
    Distance between thumb tip (4) and index tip (8), normalized by
    wrist(0)-to-index_mcp(5) distance so it works at any distance
    from the camera (same trick used in the 3D flower project).
    """
    thumb = landmarks[4]
    index = landmarks[8]
    wrist = landmarks[0]
    mcp = landmarks[5]

    ref_dist = math.hypot((mcp.x - wrist.x) * w, (mcp.y - wrist.y) * h)
    if ref_dist < 1:
        return None, None, None

    raw_dist = math.hypot((thumb.x - index.x) * w, (thumb.y - index.y) * h)
    normalized = raw_dist / ref_dist  # roughly 0.15 (pinched) to ~1.6 (spread)

    thumb_px = (int(thumb.x * w), int(thumb.y * h))
    index_px = (int(index.x * w), int(index.y * h))

    return normalized, thumb_px, index_px


print("Gesture Volume Control running. Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)  # mirror, feels natural
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark
        mp_draw.draw_landmarks(
            frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS
        )

        normalized, thumb_px, index_px = calc_pinch_distance(landmarks, w, h)

        if normalized is not None:
            # Map normalized pinch distance to 0-100 volume percent.
            # Tune these two bounds if it feels too sensitive/insensitive.
            MIN_NORM, MAX_NORM = 0.15, 1.1
            target_percent = np.interp(normalized, [MIN_NORM, MAX_NORM], [0, 100])
            target_percent = float(np.clip(target_percent, 0, 100))

            # Smooth to avoid jittery volume jumps
            smoothed_vol_percent += (target_percent - smoothed_vol_percent) * 0.25

            volume_ctrl.SetMasterVolumeLevelScalar(smoothed_vol_percent / 100.0, None)

        # Synchronize Windows mute state
        if smoothed_vol_percent <= 1:
            volume_ctrl.SetMute(1, None)
        else:
            volume_ctrl.SetMute(0, None)

            # Visual feedback
            cv2.line(frame, thumb_px, index_px, (0, 255, 0), 3)
            cv2.circle(frame, thumb_px, 8, (255, 0, 180), -1)
            cv2.circle(frame, index_px, 8, (255, 0, 180), -1)

            # Volume bar
            bar_h = int(np.interp(smoothed_vol_percent, [0, 100], [400, 150]))
            cv2.rectangle(frame, (50, 150), (85, 400), (255, 255, 255), 2)
            cv2.rectangle(frame, (50, bar_h), (85, 400), (0, 255, 0), -1)
            cv2.putText(
                frame, f"{int(smoothed_vol_percent)}%", (40, 430),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
            )

    cv2.imshow("Gesture Volume Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
