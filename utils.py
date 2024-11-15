import time
import cv2
import os
from typing import Callable


def timer(func):
    """
    A decorator function to measure and print the execution time of a function.
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.4f} seconds")
        return result

    return wrapper


def pad_with_zeros(num: int, count: int):
    """
    Pads the input number with leading zeros to ensure a string of a specified length.
    """
    return str(num).zfill(count)


BinaryOperation = Callable[[int, int], int]


def create_video_from_images(directory: str, output_video: str, fps: int = 30):
    images = sorted(
        [img for img in os.listdir(directory) if img.lower().endswith(".jpg")]
    )

    if not images:
        print("No JPG images found in the directory.")
        return

    first_image_path = os.path.join(directory, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape
    output_file_name = f"{directory}/{output_video}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    video = cv2.VideoWriter(output_file_name, fourcc, fps, (width, height))

    for image in images:
        img_path = os.path.join(directory, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    video.release()
    print(f"Video created successfully: '{output_file_name}'")
