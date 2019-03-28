import math


def yaw(eye, center):
    eye_x, eye_z = eye[0], eye[2]
    center_x, center_z = center[0], center[2]
    eye_direction = (center_x - eye_x, center_z - eye_z)
    frontward_direction = (0, 1)
    norm_eye = math.sqrt(eye_direction[0] * eye_direction[0] +
                         eye_direction[1] * eye_direction[1])
    cos = (eye_direction[0] * frontward_direction[0] +
           eye_direction[1] * frontward_direction[1]) / (norm_eye * 1.0)
    rad = math.acos(cos)
    if eye_direction[0] < 0:
        rad = -rad
    return rad


def pitch(eye, center):
    eye_direction = (center[0] - eye[0], center[1] - eye[1],
                     center[2] - eye[2])
    radius = math.sqrt(eye_direction[0]**2 + eye_direction[2]**2)
    rad = math.atan(eye_direction[1] / (radius + 1e-16))
    return rad