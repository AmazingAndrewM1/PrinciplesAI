import argparse
import json
from PIL import Image
import numpy
import itertools
from matplotlib import pyplot as plt

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

def get_masked_image(path: str):
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

def generate_probability_gradient(mask_image: Image):
    # Make the center of the walkways more likely than the edges with BFS and do it in a relatively smooth way

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

    return 1.0 / numpy.sum(distance_gradient) * distance_gradient

def generate_samples(probabilities: numpy.ndarray, num_samples: int):
    # Select samples based on the probabilities from with rng.choice()

    rng = numpy.random.default_rng()
    samples = rng.choice(
        probabilities.shape[0] * probabilities.shape[1], 
        num_samples,
        True,
        probabilities.ravel()
    )

    x, y = numpy.divmod(samples, probabilities.shape[0])
    return numpy.column_stack((x, y))

def main():
    parser = argparse.ArgumentParser(prog="Particle Filter")
    parser.add_argument("--mask", type=str, required=True, help="path of image mask with white pixels indicating walkable regions and black pixels representing non-walkable regions")
    parser.add_argument("--json", type=str, required=True, help="json file with a field for pixel_per_meter")
    args = parser.parse_args()

    mask_path = args.mask
    mask_image: Image = get_masked_image(mask_path)

    json_path = args.json
    # pixels_per_meter: float = get_pixels_per_meter(json_path)

    gradient: numpy.ndarray = generate_probability_gradient(mask_image)

    samples = generate_samples(gradient, 128)
    plt.imshow(mask_image)
    plt.scatter(samples[:, 0], samples[:, 1], marker="+", label="Samples")
    plt.legend()
    plt.show()

    






if __name__ == "__main__":
    main()