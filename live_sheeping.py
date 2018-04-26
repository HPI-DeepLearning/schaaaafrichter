import argparse
import time

import cv2

from sheeping.camera import Camera
from sheeping.sheep_localizer import SheepLocalizer


FPS_FONT = cv2.FONT_HERSHEY_SIMPLEX


def print_fps(image, fps):
    image_height, image_width, _ = image.shape
    fps_text = "{fps:.2} FPS".format(fps=fps)
    text_size = cv2.getTextSize(fps_text, FPS_FONT, 1, 1)[0]
    text_start = image_width - text_size[0], text_size[1]
    cv2.putText(image, fps_text, text_start, FPS_FONT, 1, (0, 255, 0), bottomLeftOrigin=False)
    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the HPI Sheep")
    parser.add_argument("model_file", help="path to saved model")
    parser.add_argument("log_file", help="path to log file that has been used to train model")
    parser.add_argument("-c", "--camera", type=int, default=0, help="camera to use")
    parser.add_argument("-g", "--gpu", type=int, default=-1, help="id of gpu to use")

    args = parser.parse_args()

    camera = Camera(camera_id=args.camera)
    localizer = SheepLocalizer(args.model_file, args.log_file, args.gpu)

    with camera:
        while True:
            frame = camera.get_frame()
            frame = cv2.flip(frame, 1)
            start_time = time.time()
            frame = localizer.localize(frame)
            end_time = time.time()
            fps = 1.0 / (end_time - start_time)
            frame = print_fps(frame, fps)
            cv2.imshow('sheeper', frame)
            if cv2.waitKey(1) == 27:
                break  # quit with ESC

    cv2.destroyAllWindows()
