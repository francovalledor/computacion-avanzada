from os import makedirs
from PIL import Image
from utils import pad_with_zeros, timer, BinayOperation

DEFUALT_OUTPUT_DIR = "result"
DEFAULT_VIDEO_DURATION = 4
DEFAULT_FRAMES_RATE = 24


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
def run(
    image_path1: str,
    image_path2: str,
    duration=DEFAULT_VIDEO_DURATION,
    frames_rate=DEFAULT_FRAMES_RATE,
    output_dir=DEFUALT_OUTPUT_DIR,
):
    total_frames_count = frames_rate * duration
    max_digit_length = len(str(total_frames_count))

    def save_image(image: Image, index: int):
        name = f"{output_dir}/{pad_with_zeros(index, max_digit_length)}.jpg"
        image.save(name)

    # START PROCESSING
    image1 = load_image(image_path1)
    image2 = load_image(image_path2)

    if image1.size != image2.size:
        print("Images must have the same size")
        return None

    pixels1 = image1.load()
    pixels2 = image2.load()
    size = image1.size

    # Create directory if it doesn't exist
    makedirs(output_dir, exist_ok=True)

    for i in range(total_frames_count):
        percent = i / total_frames_count

        result_image = process_images(pixels1, pixels2, size, fade_operation(percent))
        save_image(result_image, i)


image_path1 = "image1.jpg"
image_path2 = "image2.jpg"

run(image_path1, image_path2)
