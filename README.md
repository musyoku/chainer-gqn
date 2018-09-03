# :construction: Work in Progress :construction:

# Neural scene representation and rendering

[https://deepmind.com/blog/neural-scene-representation-and-rendering/](https://deepmind.com/blog/neural-scene-representation-and-rendering/)

**Todo**

- [x] OpenGL Renderer
    - [ ] Shading
    - [ ] Texture
    - [ ] Headless rendering
- [x] Implement GQN
- [x] Implement training loop
- [ ] Hyperparameter search (It takes 2 weeks to run a training)
- [ ] Debug

Current training progress:

:thinking::thinking::thinking:

![https://thumbs.gfycat.com/OffbeatCoordinatedArcherfish-size_restricted.gif](https://thumbs.gfycat.com/OffbeatCoordinatedArcherfish-size_restricted.gif)

```
Iteration 235 - loss: nll_per_pixel: 0.566455 mse: 0.004107 kld: 15.464150 - lr: 4.1628e-05 - sigma_t: 0.700000 - step: 1037760 - elapsed_time: 2.135 min
```

# Requirements

- pybind11
- Python 3
- OpenGL 4.5
- GLFW 3
- Ubuntu
- C++14 (gcc-6)
- Chainer 4

# Installation

**GLFW**

```
sudo apt install libglfw3-dev
```

**pybind11**

```
pip3 install pybind11 --user
```

**Renderer**

```
cd three
make
```

**imgplot**

```
cd imgplot
make
```

**Chainer**

```
pip3 install chainer cupy h5py
```

# Dataset

There are two choices:

- Run `construct_datasets/shepard_matzler.py` to generate observations with your own scene settings.
- Convert [the official dataset](https://github.com/deepmind/gqn-datasets) to NumPy array by [gqn-datasets-translator](https://github.com/musyoku/gqn-datasets-translator).


# Training
# Experimental Results