import time
from mpi4py import MPI
from os import makedirs
from PIL import Image
from utils import pad_with_zeros
import argparse

DEFAULT_OUTPUT_DIR = "result"
DEFAULT_VIDEO_DURATION = 4
DEFAULT_FRAMES_RATE = 24


def load_image(path):
    try:
        image = Image.open(path).convert("RGB")
        return image
    except Exception as e:
        raise Exception(f"Error loading image: {e}")


def process_images(pixels1, pixels2, size, index, total_frames_count):
    operation = fade_operation(index / total_frames_count)

    result_pixels = []

    for i in range(len(pixels1)):
        (r1, g1, b1) = pixels1[i]
        (r2, g2, b2) = pixels2[i]

        result_pixels.append(
            (
                operation(r1, r2),
                operation(g1, g2),
                operation(b1, b2),
            )
        )

    result_image = Image.new("RGB", size)
    result_image.putdata(result_pixels)
    return result_image


def fade_operation(percent: float):
    def operation(p1: int, p2: int) -> int:
        return int(p1 * percent + p2 * (1 - percent))

    return operation


def save_image(image: Image, index: int, output_dir: str, max_digit_length: int):
    name = f"{output_dir}/{pad_with_zeros(index, max_digit_length)}.jpg"
    image.save(name)


def run(image_path1: str, image_path2: str, duration, frames_rate, output_dir):
    comm = MPI.COMM_WORLD
    my_id = comm.Get_rank()
    total_processes = comm.Get_size()

    total_frames_count = frames_rate * duration
    max_digit_length = len(str(total_frames_count))

    # Checks and loading images
    if my_id == 0:
        start_time = time.time()

        # Create directory if it doesn't exist
        makedirs(output_dir, exist_ok=True)

        image1 = load_image(image_path1)
        image2 = load_image(image_path2)

        if image1.size != image2.size:
            raise Exception("Images must have the same size")

        # Convert pixel data to a serializable format (list)
        pixels1 = list(image1.getdata())
        pixels2 = list(image2.getdata())
        image_size = image1.size

    else:
        pixels1 = None
        pixels2 = None
        image_size = None

    # Broadcast the image size to all processes
    image_size = comm.bcast(image_size, root=0)

    # Scatter pixel data to all processes
    pixels1 = comm.bcast(pixels1, root=0)
    pixels2 = comm.bcast(pixels2, root=0)

    results = []

    # Start processing each process
    for i in range(my_id, total_frames_count, total_processes):
        image_i = process_images(pixels1, pixels2, image_size, i, total_frames_count)
        results.append((i, image_i))

    # Send results back to process 0
    if my_id != 0:
        comm.send(results, dest=0)
    else:
        for index, result_image in results:
            save_image(result_image, index, output_dir, max_digit_length)

        # Receive results from other processes
        for source in range(1, total_processes):
            try:
                results = comm.recv(source=source)

                for index, image in results:
                    save_image(image, index, output_dir, max_digit_length)
            except Exception:
                break

    comm.Barrier()

    if my_id == 0:
        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.4f} seconds")


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
