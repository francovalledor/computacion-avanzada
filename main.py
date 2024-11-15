from os import makedirs
from PIL import Image
from utils import create_video_from_images, pad_with_zeros, timer, BinaryOperation
import argparse
import multiprocessing

DEFAULT_OUTPUT_DIR = "result"
DEFAULT_VIDEO_DURATION = 4
DEFAULT_FRAMES_RATE = 24


def save_image(image: Image, name: str):
    image.save(name)


def load_image(path):
    try:
        image = Image.open(path).convert("RGB")
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def process_images(pixels1, pixels2, size, operation: BinaryOperation):
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
    image_path1: str,
    image_path2: str,
    output_name: str,
    percent: float,
    size,
):
    operation = fade_operation(percent)

    image1 = load_image(image_path1)
    image2 = load_image(image_path2)
    pixels1 = image1.load()
    pixels2 = image2.load()

    result_image = process_images(pixels1, pixels2, size, operation)
    save_image(result_image, output_name)


@timer
def run(
    image_path1: str,
    image_path2: str,
    duration,
    frames_rate,
    output_dir,
    num_processes,
):
    def get_image_name(index: int):
        return f"{output_dir}/{pad_with_zeros(index, max_digit_length)}.jpg"

    total_frames_count = frames_rate * duration
    max_digit_length = len(str(total_frames_count))

    print(f"using {num_processes} processes")
    print(f"{total_frames_count} to be created. {duration}s at {frames_rate} fps")
    print(f"Output dir: '{output_dir}'")

    # START PROCESSING
    image1 = load_image(image_path1)
    image2 = load_image(image_path2)

    if image1.size != image2.size:
        print("Images must have the same size")
        return None

    size = image1.size

    # Create directory if it doesn't exist
    makedirs(output_dir, exist_ok=True)

    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(
            worker,
            [
                (
                    image_path1,
                    image_path2,
                    get_image_name(index),
                    index / total_frames_count,
                    size,
                )
                for index in range(total_frames_count)
            ],
        )

    create_video_from_images(output_dir, "output", frames_rate)


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

    parser.add_argument(
        "--processes",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Duration of the video in seconds.",
    )

    args = parser.parse_args()

    run(
        args.image1,
        args.image2,
        output_dir=args.output_dir,
        frames_rate=args.frames_per_second,
        duration=args.duration,
        num_processes=args.processes,
    )
