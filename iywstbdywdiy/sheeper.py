import argparse
import os
import json
import random

from tqdm import tqdm
from viewer import Viewer
from PIL import Image, ImageTk


class Generator:
    def __init__(self, output, searchpath=None):
        self.stamps = []
        self.output = output
        self.searchpath = searchpath
        os.makedirs(self.output, exist_ok=True)

        self.i = 0
        self.info = []

    def load_stamps(self, stamps):
        self.stamps = []
        for stamp_path in stamps:
            self.stamps.append(Image.open(stamp_path))

    def get_data_for(self, image_path):
        file, _ = os.path.splitext(os.path.basename(image_path))
        directory = os.path.dirname(image_path)

        data_dir = directory
        if self.searchpath is not None:
            data_dir = self.searchpath

        data_file = os.path.join(data_dir, "{}.json".format(file))

        content = []
        with open(data_file) as data_handle:
            content = json.load(data_handle)
        return content

    def process_image(self, image_path):
        image = Image.open(image_path).convert(mode="RGBA")

        bounding_boxes = self.get_data_for(image_path)

        self.make_image(image)
        for bounding_box in bounding_boxes:
            for stamp in self.stamps:
                self.make_image(image, [bounding_box], [stamp])

        for nr_bboxes in range(2, min(len(bounding_boxes), 3)):
            bbox_indices = list(range(len(bounding_boxes)))
            for i in range(len(bounding_boxes) - nr_bboxes):
                bbox_indices.remove(random.choice(bbox_indices))
            stamps = []
            for i in range(nr_bboxes):
                stamps.append(random.choice(self.stamps))
            self.make_image(image, [bounding_boxes[i] for i in bbox_indices], stamps)

    def make_image(self, image, bounding_boxes=[], stamps=[]):
        output_path = self.get_next_output_path()
        self.info.append({"image": output_path, "bounding_boxes": bounding_boxes})

        out = image

        for i, bbox in enumerate(bounding_boxes):
            x1, y1, x2, y2 = bbox

            width = x2 - x1
            height = y2 - y1

            resized_to_bb = stamps[i].resize((width, height), Image.ANTIALIAS)

            layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
            layer.paste(resized_to_bb, box=(x1, y1))
            out = Image.alpha_composite(out, layer)

        out.convert("RGB").save(output_path)
        self.save_list()

    def get_next_output_path(self):
        self.i += 1
        return os.path.join(self.output, "{:06d}.jpg".format(self.i - 1))

    def save_list(self):
        with open(os.path.join(self.output, "info.json"), "w") as list_file:
            list_file.write(json.dumps(self.info, indent=2))


def main(args):
    g = Generator(args.output, args.searchpath)
    g.load_stamps(args.stamps)

    for image_path in tqdm(args.image):
        g.process_image(image_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Paste a number of images into other images with bounding boxes")
    parser.add_argument("image", nargs="+", help="image files to paste into")
    parser.add_argument("--searchpath", default=None, help="path to search for corresponding json files")
    parser.add_argument("--stamps", default=["sheep.png"], nargs="+", help="path to search for images to paste")
    parser.add_argument("--output", default="output", help="output directory")

    main(parser.parse_args())
