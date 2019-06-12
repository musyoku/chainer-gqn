# :construction: Work in Progress :construction:

# Neural scene representation and rendering

[https://deepmind.com/blog/neural-scene-representation-and-rendering/](https://deepmind.com/blog/neural-scene-representation-and-rendering/)

# Requirements

- Python 3
- h5py
- Chainer
    - `pip3 install chainer`
- CuPy
    - `pip3 install cupy-cuda100` for CUDA 10.0
    - `pip3 install cupy-cuda91` for CUDA 9.1

Also you need the followings for visualization:

- ffmpeg
    - `sudo apt install ffmpeg`
- imagemagick
    - `sudo apt install imagemagick`

**Current training progress**

![figure](https://thumbs.gfycat.com/RevolvingAntiqueCopepod.webp)

![figure](https://thumbs.gfycat.com/ThoughtfulQuerulousGlobefish.webp)

# Network Architecture

![gqn_conv_draw](https://user-images.githubusercontent.com/15250418/50375239-ad31bb00-063d-11e9-9c1b-151c18dc265d.png)

![gqn_representation](https://user-images.githubusercontent.com/15250418/50375240-adca5180-063d-11e9-8b2a-fb2c3995bc33.png)

# Dataset

## deepmind/gqn-datasets

Datasets used to train GQN in the paper are available to download.

https://github.com/deepmind/gqn-datasets

You need to convert `.tfrecord` files to HDF5 `.h5` format before starting training.

https://github.com/musyoku/gqn-datasets-translator

## gqn-dataset-renderer

I am working on a OpenGL/CUDA renderer for rendering GQN dataset.

https://github.com/musyoku/gqn-dataset-renderer

- **Shepard-Metzler**

![shepard_matzler](https://user-images.githubusercontent.com/15250418/54495487-92fb3680-4927-11e9-83be-125b669701db.gif)

- **Rooms**

![rooms_rotate_object](https://user-images.githubusercontent.com/15250418/54522553-e5346a00-49b0-11e9-8149-221a18e68a05.gif)

- **MNIST Dice**

![mnist_dice](https://user-images.githubusercontent.com/15250418/54581222-119ec380-4a4f-11e9-960b-db679e33723f.gif)
