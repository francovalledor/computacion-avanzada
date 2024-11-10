from os import makedirs
from PIL import Image
from utils import pad_with_zeros, timer, BinayOperation
import argparse
import multiprocessing

DEFAULT_OUTPUT_DIR = "result"
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


def worker(
    index: int, image_path1: str, image_path2: str, total_frames_count: int, size
):
    percent = index / total_frames_count
    operation = fade_operation(percent)

    image1 = load_image(image_path1)
    image2 = load_image(image_path2)
    pixels1 = image1.load()
    pixels2 = image2.load()

    result_image = process_images(pixels1, pixels2, size, operation)
    return result_image


@timer
def run(
    image_path1: str,
    image_path2: str,
    duration,
    frames_rate,
    output_dir,
):
    def save_image(image: Image, index: int):
        name = f"{output_dir}/{pad_with_zeros(index, max_digit_length)}.jpg"
        image.save(name)

    total_frames_count = frames_rate * duration
    max_digit_length = len(str(total_frames_count))
    results = [None] * total_frames_count

    # START PROCESSING
    image1 = load_image(image_path1)
    image2 = load_image(image_path2)

    if image1.size != image2.size:
        print("Images must have the same size")
        return None

    size = image1.size

    # Create directory if it doesn't exist
    makedirs(output_dir, exist_ok=True)

    num_processes = multiprocessing.cpu_count()

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(
            worker,
            [
                (i, image_path1, image_path2, total_frames_count, size)
                for i in range(total_frames_count)
            ],
        )

    for i in range(total_frames_count):
        save_image(results[i], i + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process two images and create a transition."
    )

    parser.add_argument("image1", type=str, help="Path to the first image.")
    parser.add_argument("image2", type=str, help="Path to the second image.")

    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the output images.",
    )

    parser.add_argument(
        "--frames_per_second",
        type=int,
        default=DEFAULT_FRAMES_RATE,
        help="Number of frames per second.",
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=DEFAULT_VIDEO_DURATION,
        help="Duration of the video in seconds.",
    )

    args = parser.parse_args()

    run(
        args.image1,
        args.image2,
        output_dir=args.output_dir,
        frames_rate=args.frames_per_second,
        duration=args.duration,
    )
