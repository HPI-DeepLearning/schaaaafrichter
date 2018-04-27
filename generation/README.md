# Scripts to generate training data

First use `python generation/annotate.py` to create define bounding boxes for a set of background images (see the [data folder](../data) for how to get background images).

Then use `python generation/sheeper.py` to paste other images into the defined bounding boxes.

Alternatively just use pregenerated data (see the [data folder](../data)).
