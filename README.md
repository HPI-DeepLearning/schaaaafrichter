# schaaaafrichter
Schaafrichter 2.0

## inference

### with the live usb application

Execute this on your host, to allow docker to connect to your X server:
```
xhost +local:docker
```

Build the container first:
```
docker build -t sheep .
```
If you use CUDA with a version earlier than 9.1, specify the corresponding docker image (see [this list](https://hub.docker.com/r/nvidia/cuda/) for available options).
For example, for CUDA 8 and CUDNN 6 use the following instead:
```
docker build -t sheep --build-arg FROM_IMAGE=nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04 .
```
Afterwards run the container:
```
docker run \
    --rm \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY \
    --device=/dev/video0:/dev/video0 \
    -it \
    --volume /absolute/path/to/data:/app/data \
    sheep
```

Inside the docker container run something like:
```
python3 live_sheeping.py data/trained_model data/log
```
