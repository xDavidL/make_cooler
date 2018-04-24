import math
import sys
import face_recognition
from PIL import Image, ImageDraw

def average(points):
    avg_x = 0
    avg_y = 0
    for x, y in points:
        avg_x += x
        avg_y += y
    num_points = len(points)
    return (avg_x / num_points, avg_y / num_points)


def angle_between(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    adjacent = x2 - x1
    opposite = y2 - y1
    angle = math.degrees(math.tan(opposite / adjacent))
    if opposite < 0:
        return angle
    return 360 - angle


def distance_between(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    adjacent = x2 - x1
    opposite = y2 - y1
    return math.sqrt(adjacent ** 2 + opposite ** 2)


def fdiv(x, y):
    return math.floor(x / y)


def main():
    image_path = sys.argv[1]
    save_path = sys.argv[2]
    glasses_path = "images/deal-with-it.jpg"
    glasses_left = (256, 103)  # center of the left lense
    glasses_right = (768, 103)
    output_file = ""
    image = face_recognition.load_image_file(image_path)
    facial_features = face_recognition.face_landmarks(image)

    eye_angles = []
    eye_dists = []
    glasses_scales = []
    left_eye_pos = []
    right_eye_pos = []
    glasses_dist = distance_between(glasses_left, glasses_right)
    for face in facial_features:
        left_eye = average(face["left_eye"])
        right_eye = average(face["right_eye"])
        eye_angles.append(angle_between(left_eye, right_eye))
        eye_dist = distance_between(left_eye, right_eye)
        eye_dists.append(eye_dist)
        glasses_scales.append(glasses_dist / eye_dist)
        left_eye_pos.append(left_eye)
        right_eye_pos.append(right_eye)

    image = Image.open(image_path)
    glasses = Image.open(glasses_path)
    glasses = glasses.convert('RGBA')
    for angle, scale, pos in zip(eye_angles, glasses_scales, left_eye_pos):
        glasses_copy = glasses.copy()
        width, height = glasses_copy.size
        glasses_copy = glasses_copy.resize((fdiv(width, scale), fdiv(height, scale)))
        left_x, left_y = glasses_left
        left_x /= scale
        left_y /= scale
        # todo: update translate to include rotation (still works without it)
        translate = (math.floor(pos[0] - left_x), math.floor(pos[1] - left_y))
        glasses_copy = glasses_copy.rotate(angle, center=(left_x, left_y), expand=True)
        image.paste(glasses_copy, translate, glasses_copy)

    image.save(save_path)


if __name__ == "__main__":
    main()
