import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def calculate_finger_positions(hand_landmarks):
    finger_points = []
    for i in range(21):  # MediaPipe hand has 21 points
        point = hand_landmarks.landmark[i]
        finger_points.append([point.x, point.y, point.z])
    return np.array(finger_points)

def get_finger_states(points):
    finger_states = []
    
    # Thumb detection
    thumb_tip = points[4]
    thumb_base = points[2]
    thumb_extended = thumb_tip[0] < thumb_base[0]
    finger_states.append(thumb_extended)
    
    # Finger detection
    for finger in range(4):
        tip_id = 8 + (finger * 4)
        mid_id = 7 + (finger * 4)
        base_id = 5 + (finger * 4)
        
        finger_tip = points[tip_id]
        finger_mid = points[mid_id]
        finger_base = points[base_id]
        
        vertical_check = finger_tip[1] < finger_base[1]
        distance_check = np.linalg.norm(finger_tip - finger_base) > 0.1
        
        finger_extended = vertical_check and distance_check
        finger_states.append(finger_extended)
    
    return finger_states

def recognize_letter(finger_states, points):
    patterns = {
        'A': [False, False, False, False, False],    # Fist
        'B': [True, True, True, True, True],         # Flat hand
        'C': [True, True, True, True, True],         # Curved C
        'D': [False, True, False, False, False],     # Index up
        'E': [True, False, False, False, False],     # Thumb
        'F': [True, True, False, False, False],      # First two
        'G': [True, True, False, False, False],      # Point sideways
        'H': [False, True, True, False, False],      # Index and middle
        'I': [False, False, False, False, True],     # Pinky
        'J': [False, False, False, False, True],     # Moving pinky
        'K': [True, True, False, False, False],      # Victory pointing
        'L': [True, True, False, False, False],      # L shape
        'M': [False, True, True, True, False],       # Three down
        'N': [False, True, True, False, False],      # Two down
        'O': [True, True, True, True, True],         # O shape
        'P': [True, True, False, False, False],      # P shape
        'Q': [True, True, False, False, False],      # Q shape
        'R': [False, True, True, False, False],      # Crossed fingers
        'S': [False, False, False, False, False],    # Fist
        'T': [False, True, False, False, False],     # T shape
        'U': [False, True, True, False, False],      # Two up close
        'V': [False, True, True, False, False],      # Two up spread
        'W': [False, True, True, True, False],       # Three up
        'X': [False, True, False, False, False],     # Bent index
        'Y': [True, False, False, False, True],      # Hang loose
        'Z': [False, True, False, False, False],     # Moving index
    }

    palm_center = np.mean(points[5:9], axis=0)
    
    for symbol, pattern in patterns.items():
        if finger_states == pattern:
            if symbol in ['U', 'V']:
                finger_spread = np.linalg.norm(points[8] - points[12])
                return 'V' if finger_spread > 0.13 else 'U'
            
            elif symbol in ['M', 'N', 'W']:
                finger_spread = np.linalg.norm(points[8] - points[16])
                if finger_spread > 0.15:
                    return 'W'
                return 'M' if finger_states[3] else 'N'
            
            elif symbol in ['A', 'S', 'E']:
                thumb_pos = points[4][0] - points[2][0]
                if thumb_pos > 0.05:
                    return 'E'
                return 'S' if points[4][1] > points[2][1] else 'A'
            
            elif symbol in ['K', 'P', 'R']:
                angle = np.arctan2(points[8][1] - points[5][1], 
                                 points[8][0] - points[5][0])
                if abs(angle) > 1.2:
                    return 'K'
                return 'R' if abs(angle) < 0.3 else 'P'
            
            return symbol
    
    return '?'