import argparse
import json
from PIL import Image
import numpy
import itertools
from matplotlib import pyplot as plt
import kalman_filter
import pathlib

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DATA_FILE_PATH = pathlib.Path() / "data" / "Accelerometer.csv"

def get_masked_image(path: pathlib.Path):
    with Image.open(path) as image:
        rgb_image = image.convert("RGB")
        colors_in_image = rgb_image.getcolors(maxcolors=2)
        if colors_in_image is None or not all(color in [WHITE, BLACK] for count, color in colors_in_image):
            raise ValueError("Image contains non-white or non-black pixels")
    return rgb_image

def get_pixels_per_meter(path: str):
    with open(path, "r") as json_file:
        data = json.load(json_file)

    pixels_per_meter = float(data["pixels_per_meter"])
    return pixels_per_meter

def get_neighbors(coordinate: tuple[int, int], max_size: int):
    x, y = coordinate

    neighbors = []
    if x - 1 >= 0:
        neighbors.append((x - 1, y))
    if x + 1 < max_size:
        neighbors.append((x + 1, y))
    if y - 1 >= 0:
        neighbors.append((x, y - 1))
    if y + 1 < max_size:
        neighbors.append((x, y + 1))
    return neighbors

def generate_probability_gradient(mask_path: pathlib.Path):
    precomputed_path = pathlib.Path() / "precomputed" / "gradient_probs.npy"
    if precomputed_path.is_file():
        return numpy.load(precomputed_path)
    
    # Make the center of the walkways more likely than the edges with BFS and do it in a relatively smooth way

    mask_image: Image = get_masked_image(mask_path)
    distance_gradient = numpy.zeros(shape=mask_image.size, dtype=int)
    has_visited = [[False for _ in range(mask_image.size[1])] for _ in range(mask_image.size[0])]

    boundary_coordinates = [
        coordinate
        for coordinate in itertools.product(range(mask_image.size[0]), repeat=2)
        if mask_image.getpixel(coordinate) == WHITE and any(
            mask_image.getpixel(c) == BLACK 
            for c in get_neighbors(coordinate, mask_image.size[0])
        )
    ]

    for x, y in boundary_coordinates:
        has_visited[x][y] = True

    min_boundary_distance = 1
    coordinates = boundary_coordinates
    while len(coordinates) > 0:
        new_coordinates = []
        for coordinate in coordinates:
            distance_gradient[coordinate[0], coordinate[1]] = min_boundary_distance

            for n_coordinate in (
                n for n in get_neighbors(coordinate, mask_image.size[0]) 
                if mask_image.getpixel(n) == WHITE and not has_visited[n[0]][n[1]]
            ):
                new_coordinates.append(n_coordinate)
                has_visited[n_coordinate[0]][n_coordinate[1]] = True
                                 
        min_boundary_distance += 1
        coordinates = new_coordinates

    result = 1.0 / numpy.sum(distance_gradient) * distance_gradient
    numpy.save(precomputed_path, result)
    return result

def generate_samples(probabilities: numpy.ndarray, headings: numpy.ndarray, num_samples: int):
    # Select samples based on the probabilities from with rng.choice()

    rng = numpy.random.default_rng()
    samples = rng.choice(
        probabilities.shape[0] * probabilities.shape[1], 
        num_samples,
        True,
        probabilities.ravel()
    )
    x, y = numpy.divmod(samples, probabilities.shape[0])
    
    heading_indices = rng.choice(
        headings.shape[0],
        num_samples,
        True
    )
    return numpy.column_stack((x, y)).astype(float), heading_indices

def get_normalized_headings():
    headings = numpy.array([
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 0.0],
        [-1.0, -1.0],
        [0.0, -1.0],
        [1.0, -1.0]
    ], dtype=float)
    return headings / numpy.linalg.norm(headings, axis=1)

def main():
    parser = argparse.ArgumentParser(prog="Particle Filter")
    parser.add_argument("--mask", type=str, required=True, help="path of image mask with white pixels indicating walkable regions and black pixels representing non-walkable regions")
    parser.add_argument("--json", type=str, required=True, help="json file with a field for pixel_per_meter")
    args = parser.parse_args()

    mask_path = pathlib.Path(args.mask)
    gradient_probabilities = generate_probability_gradient(mask_path)

    json_path = args.json
    pixels_per_meter: float = get_pixels_per_meter(json_path)

    normalized_headings = get_normalized_headings()
    positions, heading_indices = generate_samples(gradient_probabilities, normalized_headings, 128)

    df_data = kalman_filter.get_trimmed_data(DATA_FILE_PATH)
    average_dt_seconds = kalman_filter.get_average_dt_seconds(df_data["time"])
    system_noise = kalman_filter.get_approximate_system_noise(0.001, average_dt_seconds)    # 0.001 is tunable to improve performance of Kalman Filter
    measurement_noise = kalman_filter.get_approximate_measurement_noise()
    kfilter = kalman_filter.KalmanFilter(average_dt_seconds, system_noise, measurement_noise)

    prev_pos = numpy.array([0.0, 0.0, 0.0], dtype=float)
    for row in df_data.loc[:, ["x", "y", "z"]].itertuples(index=False):
        datum = numpy.array((row.x, row.y, row.z), dtype=float)
        kfilter.predict()
        kfilter.update(datum)

        curr_pos = kfilter.x.ravel()[6:9]   # Only consider pos_x, pos_y, pos_z for particle filter
        displacement_imu = pixels_per_meter * (curr_pos - prev_pos)
        prev_pos = curr_pos

        # Note: Movement in the +Z 3D = +X 2D, +Y 3D = -Y 2D
        change_x_2d = displacement_imu[2]
        change_y_2d = -displacement_imu[1]

        positions[:, 0] += normalized_headings[heading_indices][:, 0] * change_x_2d
        positions[:, 1] += normalized_headings[heading_indices][:, 1] * change_y_2d
    
if __name__ == "__main__":
    main()