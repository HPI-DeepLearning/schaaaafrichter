# Schaaaafrichter

The successor of the original [Schaafrichter](https://github.com/Bartzi/schaafrichter).
Now with Advanced AI-Technolgies!

## Installation

You can build the project with or without support for a CUDA capable GPU.
Steps only applicable to one of them will be denoted (**GPU Support**/**CPU**).
Also you can either install this [in your own system](#install-on-system), or [use a docker image instead](#install-with-docker).

### Install on System

1. Make sure to install `Python 3` on your device
   - Windows: Get it [here](https://www.python.org/downloads/windows/)
   - Mac: Get it [here](https://www.python.org/downloads/mac-osx/) or use
   your favourite package manager
   - Linux: Use your favourite package manager i.e. `pacman -S python` or
   `apt install python3`
2. **GPU Support:**
   - install `CUDA`
   - install `cudnn`
3. Create a virtualenvironment
   - you can do so with `python3 -m venv --system-site-packages <path to virtualenv>`
   - If you are using Linux, we recommend that you install
   [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/)
   and organize all virtual environments with this tool, its quite neat.
   - **INFO:** Make sure to include the global site-packages that contain `Opencv` (make sure to use `--system-site-packages`)!
4. Load the virtual environment
5. Clone the repository
6. *For Ubuntu*: Install header files for alsa: `apt install libasound2-dev`
7. Install all necessary libraries:
   - **GPU Support:** `pip install -r requirements.txt`
   - **CPU:** `pip install -r requirements_cpu.txt`

### Install with Docker

1. Install `Docker`
   - Windows: Get it [here](https://www.docker.com/community-edition)
   - Mac: Get it [here](https://www.docker.com/community-edition)
   - Linux: User your favourite package manager i.e. `pacman -S docker`, or use [this guide](https://docs.docker.com/install/linux/docker-ce/ubuntu/) for Ubuntu.
   or `apt install docker`
2. **GPU Support:** In case your device has a CUDA capable GPU, you should do the following:
   - install `CUDA`
   - install `cudnn`
   - install `nvidia-docker` ([Ubuntu](https://gist.github.com/dsdenes/d9c66361df96bce3fca8f1414bb14bce),
  [Arch Like OS](https://aur.archlinux.org/packages/nvidia-docker2/))
3. Build the Docker Image:
   - **CPU:**
   ```
   docker build -t sheep --build-arg FROM_IMAGE=ubuntu:16.04 --build-arg CPU_ONLY=true .
   ```
   - **GPU Support:**
   ```
   docker build -t sheep .
   ```
   If your host system uses CUDA with a version earlier than 9.1, specify the corresponding docker image to match the configuration of your machine (see [this list](https://hub.docker.com/r/nvidia/cuda/) for available options).
   For example, for CUDA 8 and CUDNN 6 use the following instead:
   ```
   docker build -t sheep --build-arg FROM_IMAGE=nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04 .
   ```

## Training

If you want to train the Schaaafrichter you will first need a dataset.
You can either [create](generation/README.md) or [download](data/README.md)
a dataset for training the Schaaafrichter.

Once you have all necessary data, you can start the training.
The training scipt is `train.py`.
We will now quickly go over all possible command-line arguments you can/have to use:
- `dataset` path to the json file containing your training dataset
- `test_dataset` path to the json file containing the validation dataset
- `--dataset-root` specify a dataset root dir that might be different from the directory of the dataset file locations, which is used as default.
- `--model` available choices are `ssd300` and `ssd512`, you can choose which kind of model you want to train. Default is `ssd512`.
- `--batchsize` the batch size to use for training. Default is `32`.
- `--gpu` which gpu to use (e.g. `0` means your first gpu). You can also give more than one gpu id. The model will then be trained in data parallel fashion. Default is `-1`, which means run on CPU.
- `--out` specifies the output directory for the trained model and log. Default is `result`
- `--resume` specify a `trainer_snapshot` and continue training.
- `--lr` set the default learning rate for the optimizer. Default is `1e-3`.

Once you started the training, grab a coffee/tea and enjoy the rest of your day.


## Inference

Once you got a trained model, you can do inference and have fun!

### with the live usb application

#### Using Docker
Execute this on your host, to allow docker to connect to your X server (needs to be done after every system restart):
```
xhost +local:docker
```

Run the container and get a command-line:
```
nvidia-docker run \
    --rm \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    --device=/dev/video0:/dev/video0 \
    -it \
    --volume /absolute/path/to/data:/app/data \
    sheep
```
**Note:** `/absolute/path/to/data` id the path to the directory with your trained model and the log file.
You can also use: `--volume "$( readlink data -f )":/app/data`, but that does not work on Windows.


**Known errors**

If you receive an error similar to this one, you need to execute `xhost +local:docker`:
```
No protocol specified
Failed to connect to Mir: Failed to connect to server socket: No such file or directory
Unable to init server: Could not connect: Connection refused

(sheeper:1): Gtk-WARNING **: cannot open display: :1
```

#### Using Docker and Virtualenv

Run something like:
```
python3 live_sheeping.py data/models/trained_model data/models/log
```
You can also run on a gpu with `--gpu <gpu_id>`.

To generate predictions for static images instead (you can add `--gpu <gpu_id>` again):

`python image_sheeping.py data/models/trained_model data/models/log -j data/generated/test_info.json`
