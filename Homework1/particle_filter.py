import argparse
import json
from PIL import Image
import numpy
import itertools
from matplotlib import pyplot as plt
import kalman_filter
import pathlib
import imageio.v2 as iv2

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DATA_FILE_PATH = pathlib.Path() / "data" / "walking-rutgers-path" / "Accelerometer.csv"

def get_masked_image(path: pathlib.Path):
    with Image.open(path) as image:
        rgb_image = image.convert("RGB")
        colors_in_image = rgb_image.getcolors(maxcolors=2)
        if colors_in_image is None or not all(color in [WHITE, BLACK] for count, color in colors_in_image):
            raise ValueError("Image contains non-white or non-black pixels")
    return rgb_image

def get_json_data(path: pathlib.Path) -> tuple[float, numpy.ndarray]:
    with path.open("r") as json_file:
        data = json.load(json_file)

    pixels_per_meter = float(data["pixels_per_meter"])
    # heading_angle = numpy.deg2rad(float(data["heading_angle_degrees"]))

    # heading = numpy.array([numpy.cos(heading_angle), -numpy.sin(heading_angle)])    # Rembember y increases downward in image coordinates
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

def generate_probability_gradient(mask_image: Image):
    precomputed_path = pathlib.Path() / "precomputed" / "gradient_probs.npy"
    if precomputed_path.is_file():
        return numpy.load(precomputed_path)
    
    arr = numpy.asarray(mask_image)
    probs = numpy.zeros(shape=arr.shape[:-1], dtype=float)
    white_mask = numpy.all(arr == WHITE, axis=2)
    probs[white_mask] = 1.0 / numpy.count_nonzero(white_mask)
    numpy.save(precomputed_path, probs)
    return probs
    
    # # Make the center of the walkways more likely than the edges with BFS and do it in a relatively smooth way

    # distance_gradient = numpy.zeros(shape=mask_image.size, dtype=int)
    # has_visited = [[False for _ in range(mask_image.size[1])] for _ in range(mask_image.size[0])]

    # boundary_coordinates = [
    #     coordinate
    #     for coordinate in itertools.product(range(mask_image.size[0]), repeat=2)
    #     if mask_image.getpixel(coordinate) == WHITE and any(
    #         mask_image.getpixel(c) == BLACK 
    #         for c in get_neighbors(coordinate, mask_image.size[0])
    #     )
    # ]

    # for x, y in boundary_coordinates:
    #     has_visited[x][y] = True

    # min_boundary_distance = 1
    # coordinates = boundary_coordinates
    # while len(coordinates) > 0:
    #     new_coordinates = []
    #     for coordinate in coordinates:
    #         distance_gradient[coordinate[0], coordinate[1]] = min_boundary_distance

    #         for n_coordinate in (
    #             n for n in get_neighbors(coordinate, mask_image.size[0]) 
    #             if mask_image.getpixel(n) == WHITE and not has_visited[n[0]][n[1]]
    #         ):
    #             new_coordinates.append(n_coordinate)
    #             has_visited[n_coordinate[0]][n_coordinate[1]] = True
                                 
    #     min_boundary_distance += 1
    #     coordinates = new_coordinates

    # result = 1.0 / numpy.sum(distance_gradient) * distance_gradient
    # numpy.save(precomputed_path, result)
    # return result

def generate_samples(probabilities: numpy.ndarray, headings: numpy.ndarray, num_samples: int):
    # Select samples based on the probabilities from with rng.choice()

    rng = numpy.random.default_rng()
    samples = rng.choice(
        probabilities.shape[0] * probabilities.shape[1], 
        num_samples,
        True,
        probabilities.ravel()
    )
    row, col = numpy.divmod(samples, probabilities.shape[1])
    
    heading_indices = rng.choice(
        headings.shape[0],
        num_samples,
        True
    )
    return numpy.column_stack((col, row)).astype(float), heading_indices

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
    return headings / numpy.atleast_2d(numpy.linalg.norm(headings, axis=1)).T

def main():
    parser = argparse.ArgumentParser(prog="Particle Filter")
    parser.add_argument("--mask", type=str, required=True, help="path of image mask with white pixels indicating walkable regions and black pixels representing non-walkable regions")
    parser.add_argument("--json", type=str, required=True, help="json file with a field for pixel_per_meter")
    args = parser.parse_args()

    mask_path = pathlib.Path(args.mask)
    mask_image = get_masked_image(mask_path)
    gradient_probabilities = generate_probability_gradient(mask_image)

    json_path = pathlib.Path(args.json)
    pixels_per_meter = get_json_data(json_path)

    normalized_headings = get_normalized_headings()
    positions, heading_indices = generate_samples(gradient_probabilities, normalized_headings, 512)

    df_data = kalman_filter.get_trimmed_data(DATA_FILE_PATH)
    average_dt_seconds = kalman_filter.get_average_dt_seconds(df_data["time"])
    system_noise = kalman_filter.get_approximate_system_noise(0.001, average_dt_seconds)    # 0.001 is tunable to improve performance of Kalman Filter
    measurement_noise = kalman_filter.get_approximate_measurement_noise()
    position_noise = kalman_filter.get_approximate_position_noise()
    kfilter = kalman_filter.KalmanFilter(average_dt_seconds, system_noise, measurement_noise)

    frames = []
    rng = numpy.random.default_rng()
    prev_pos = numpy.array([0.0, 0.0, 0.0], dtype=float)
    next_second = 0
    for row in df_data.itertuples(index=False):
        datum = numpy.array((row.x, row.y, row.z), dtype=float)
        kfilter.predict()
        kfilter.update(datum)

        curr_pos = kfilter.x.ravel()[6:9]   # Only consider pos_x, pos_y, pos_z for particle filter
        displacement_imu = pixels_per_meter * (curr_pos - prev_pos)
        prev_pos = curr_pos

        turning_mask = rng.random(size=heading_indices.shape[0]) < 0.1
        counterclockwise_mask = rng.choice([True, False], size=heading_indices.shape[0], replace=True, shuffle=False)
        heading_indices[turning_mask & counterclockwise_mask] += 1
        heading_indices[turning_mask & ~counterclockwise_mask] -= 1
        heading_indices %= normalized_headings.shape[0]

        # Note: Movement in the +Z 3D = +X 2D, +X 3D = -Y 2D
        change_2d = numpy.array((displacement_imu[2], -displacement_imu[0]), dtype=float)
        positions += normalized_headings[heading_indices] * numpy.atleast_2d(change_2d)
        
        positions[:, 0] += rng.normal(0.0, position_noise[0], size=positions.shape[0])
        positions[:, 1] += rng.normal(0.0, position_noise[1], size=positions.shape[0])

        if row.seconds_elapsed > next_second:
            fig = plt.figure(figsize=(5, 5))
            plt.xlim((0.0, mask_image.size[0]))
            plt.ylim((mask_image.size[1], 0.0))
            plt.imshow(mask_image, cmap="gray")
            plt.scatter(positions[:, 0], positions[:, 1], marker="+")

            # Thanks to https://gist.github.com/samuelsmal/432db47096cbf5e141bff37d2f367ab1 to
            # show how to make a matplotlib plot and save it in memory

            fig.canvas.draw()
            frame = numpy.array(fig.canvas.renderer._renderer)
            frames.append(frame)
            plt.close()

            next_second += 1

        # # Resampling time
        # # 1) Find probabilities of points

        point_probabilities: numpy.ndarray = gradient_probabilities[positions[:, 0].astype(int), positions[:, 1].astype(int)]
        point_probabilities *= 1.0 / point_probabilities.sum()
        selected_indices = rng.choice(
            positions.shape[0], 
            size=positions.shape[0], 
            replace=True, 
            p=point_probabilities
        )

        positions = positions[selected_indices]
        heading_indices = heading_indices[selected_indices]
    
    iv2.mimsave("./animation/animation.gif", frames, duration=5.0, loop=0)
    
if __name__ == "__main__":
    main()