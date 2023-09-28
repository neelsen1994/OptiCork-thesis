FROM tensorflow/tensorflow:latest-gpu-jupyter

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsm6 libxext6

RUN pip install numpy  pandas scipy Pillow cython matplotlib scikit-image tensorflow>=2.0.0 keras opencv-python h5py imgaug torchvision IPython[all]



