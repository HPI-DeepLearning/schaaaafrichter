import argparse

import cv2

from sheeping.camera import Camera
from sheeping.sheep_localizer import SheepLocalizer

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
            frame = localizer.localize(frame)
            cv2.imshow('sheeper', frame)
            if cv2.waitKey(1) == 27:
                break  # quit with ESC

    cv2.destroyAllWindows()
