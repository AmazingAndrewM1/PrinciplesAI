import argparse
import json
from PIL import Image

def get_masked_image(path: str):
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
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

def generate_probability_gradient(mask_image: Image):
    print(mask_image._check_size())
    pass

def main():
    parser = argparse.ArgumentParser(prog="Particle Filter")
    parser.add_argument("--mask", type=str, required=True, help="path of image mask with white pixels indicating walkable regions and black pixels representing non-walkable regions")
    parser.add_argument("--json", type=str, required=True, help="json file with a field for pixel_per_meter")
    args = parser.parse_args()

    mask_path = args.mask
    mask_image: Image = get_masked_image(mask_path)

    json_path = args.json
    pixels_per_meter: float = get_pixels_per_meter(json_path)

    generate_probability_gradient(mask_image)



if __name__ == "__main__":
    main()