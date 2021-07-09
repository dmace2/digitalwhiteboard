import mediapipe as mp
import numpy as np


def finger_angles(landmarks):
    mp_hands = mp.solutions.hands

    x = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
    y = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    z = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z

    ftip = np.array([x, y, z])

    x = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x
    y = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
    z = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].z

    mcp = np.array([x, y, z])

    x = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x
    y = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    z = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].z

    pip = np.array([x, y, z])

    vec1 = ftip - mcp
    vec2 = pip - mcp

    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)

    indang = np.arccos(np.dot(vec1, vec2))

    x = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x
    y = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    z = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].z

    ftip = np.array([x, y, z])

    x = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x
    y = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
    z = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z

    mcp = np.array([x, y, z])

    x = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x
    y = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    z = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].z

    pip = np.array([x, y, z])

    vec1 = ftip - mcp
    vec2 = pip - mcp

    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)

    midang = np.arccos(np.dot(vec1, vec2))

    x = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x
    y = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
    z = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].z

    ftip = np.array([x, y, z])

    x = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x
    y = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y
    z = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].z

    mcp = np.array([x, y, z])

    x = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x
    y = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y
    z = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].z

    pip = np.array([x, y, z])

    vec1 = ftip - mcp
    vec2 = pip - mcp

    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)

    rngang = np.arccos(np.dot(vec1, vec2))

    x = landmarks.landmark[mp_hands.HandLandmark.PINKY_FINGER_TIP].x
    y = landmarks.landmark[mp_hands.HandLandmark.PINKY_FINGER_TIP].y
    z = landmarks.landmark[mp_hands.HandLandmark.PINKY_FINGER_TIP].z

    ftip = np.array([x, y, z])

    x = landmarks.landmark[mp_hands.HandLandmark.PINKY_FINGER_MCP].x
    y = landmarks.landmark[mp_hands.HandLandmark.PINKY_FINGER_MCP].y
    z = landmarks.landmark[mp_hands.HandLandmark.PINKY_FINGER_MCP].z

    mcp = np.array([x, y, z])

    x = landmarks.landmark[mp_hands.HandLandmark.PINKY_FINGER_PIP].x
    y = landmarks.landmark[mp_hands.HandLandmark.PINKY_FINGER_PIP].y
    z = landmarks.landmark[mp_hands.HandLandmark.PINKY_FINGER_PIP].z

    pip = np.array([x, y, z])

    vec1 = ftip - mcp
    vec2 = pip - mcp

    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)

    pnkang = np.arccos(np.dot(vec1, vec2))

    return [indang, midang, rngang, pnkang]
