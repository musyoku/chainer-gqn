# :construction: Work in Progress :construction:

# Neural scene representation and rendering

[https://deepmind.com/blog/neural-scene-representation-and-rendering/](https://deepmind.com/blog/neural-scene-representation-and-rendering/)

**Todo**

- [x] Implement GQN
- [x] Implement training loop
- [ ] Hyperparameter search
- [ ] Debug

Current training progress:

:thinking::thinking::thinking:

![https://thumbs.gfycat.com/OffbeatCoordinatedArcherfish-size_restricted.gif](https://thumbs.gfycat.com/OffbeatCoordinatedArcherfish-size_restricted.gif)

```
Iteration 235 - loss: nll_per_pixel: 0.566455 mse: 0.004107 kld: 15.464150 - lr: 4.1628e-05 - sigma_t: 0.700000 - step: 1037760 - elapsed_time: 2.135 min
```

# Requirements

- Python 3
- Chainer 4

# Dataset

https://github.com/musyoku/gqn-dataset-renderer

- **Shepard-Matzler**

![shepard_matzler](https://user-images.githubusercontent.com/15250418/47383748-53496d80-d740-11e8-8db8-e7a25bd1ad5c.gif)

- **Rooms**

![anim](https://user-images.githubusercontent.com/15250418/47347087-7e54a280-d6e9-11e8-93db-47dd2b4efaea.gif)

- **MNIST Dice**

![rooms](https://user-images.githubusercontent.com/15250418/47368004-ce4c5d00-d71b-11e8-9834-bf87b128892b.gif)