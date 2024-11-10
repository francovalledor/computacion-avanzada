from mpi4py import MPI
from os import makedirs
from PIL import Image
from utils import pad_with_zeros, timer, BinayOperation
import argparse

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


def save_image(image: Image, index: int, output_dir: str, max_digit_length: int):
    name = f"{output_dir}/{pad_with_zeros(index, max_digit_length)}.jpg"
    image.save(name)


@timer
def run(image_path1: str, image_path2: str, duration, frames_rate, output_dir):
    total_frames_count = frames_rate * duration
    max_digit_length = len(str(total_frames_count))

    # Create directory if it doesn't exist
    makedirs(output_dir, exist_ok=True)

    # Load images only in the root process
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        image1 = load_image(image_path1)
        image2 = load_image(image_path2)

        if image1.size != image2.size:
            print("Images must have the same size")
            return None

        pixels1 = image1.load()
        pixels2 = image2.load()
        image_dimensions = image1.size

        frames = [
            (pixels1, pixels2, image_dimensions, fade_operation(i / total_frames_count))
            for i in range(total_frames_count)
        ]
    else:
        frames = None

    # Broadcast the frames to all processes
    frames = comm.bcast(frames, root=0)

    # Each process works on a subset of frames
    for i in range(rank, total_frames_count, size):
        pixels1, pixels2, size, operation = frames[i]
        image = process_images(pixels1, pixels2, size, operation)
        comm.send((i, image), dest=0)

    # Root process collects results
    if rank == 0:
        for _ in range(size):
            index, image = comm.recv(source=MPI.ANY_SOURCE)
            save_image(image, index + 1, output_dir, max_digit_length)

        print("All done!")


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
