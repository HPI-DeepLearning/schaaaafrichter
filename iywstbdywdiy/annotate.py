import argparse

from viewer import Viewer


def main(args):
    viewer = Viewer()
    viewer.images = args.image
    viewer.master.title('Viewer')
    viewer.mainloop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Create simple bounding box annotations for a simple neural network")
    parser.add_argument("image", nargs="+", help="image files to annotate")

    main(parser.parse_args())
