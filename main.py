from os import makedirs
from PIL import Image
from numpy import number
from utils import timer, BinayOperation

DEFUALT_OUTPUT_DIR = "result"

VIDEO_DURATION = 4
FRAMES_PER_SECOND = 24
TOTAL_FRAMES_COUNT = FRAMES_PER_SECOND * VIDEO_DURATION
DIGITS_COUNT = len(str(TOTAL_FRAMES_COUNT))


def get_frame_name(index: number):
    padded_index = str(index).zfill(DIGITS_COUNT)

    return f"{DEFUALT_OUTPUT_DIR}/{padded_index}.jpg"


def save_image(image: Image, index: int):
    image.save(get_frame_name(index))


def load_image(path):
    try:
        image = Image.open(path).convert("RGB")
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def process_images(pixels1, pixels2, size, operation: BinayOperation):
    result_image = Image.new("RGB", size)
    result_pixels = result_image.load()

    (width, height) = size

    for i in range(width):
        for j in range(height):
            r1, g1, b1 = pixels1[i, j]
            r2, g2, b2 = pixels2[i, j]
            result_pixels[i, j] = (
                operation(r1, r2),
                operation(g1, g2),
                operation(b1, b2),
            )

    return result_image


def fade_operation(percent: float):
    def operation(p1: int, p2: int) -> int:
        return int(p1 * percent + p2 * (1 - percent))

    return operation


@timer
def run(image_path1: str, image_path2: str):
    image1 = load_image(image_path1)
    image2 = load_image(image_path2)

    if image1.size != image2.size:
        print("Images must have the same size")
        return None

    pixels1 = image1.load()
    pixels2 = image2.load()
    size = image1.size

    makedirs(DEFUALT_OUTPUT_DIR, exist_ok=True)

    for i in range(TOTAL_FRAMES_COUNT):
        percent = i / TOTAL_FRAMES_COUNT

        result_image = process_images(pixels1, pixels2, size, fade_operation(percent))
        save_image(result_image, i)


image_path1 = "image1.jpg"
image_path2 = "image2.jpg"

run(image_path1, image_path2)
