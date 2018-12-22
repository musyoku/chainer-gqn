# :construction: Work in Progress :construction:

# Neural scene representation and rendering

[https://deepmind.com/blog/neural-scene-representation-and-rendering/](https://deepmind.com/blog/neural-scene-representation-and-rendering/)

**Todo**

- [x] Implement GQN
- [x] Implement training loop
- [ ] Hyperparameter search

Current training progress:

:thinking::thinking::thinking:

**Shepard-Matzler 64x64**

![shepard_matzler](https://thumbs.gfycat.com/ForthrightBrokenCanadagoose.webp)

https://gfycat.com/ForthrightBrokenCanadagoose

![shepard_matzler](https://thumbs.gfycat.com/MajorSeriousKittiwake.webp)

https://gfycat.com/MajorSeriousKittiwake

![shepard_matzler](https://thumbs.gfycat.com/PartialYellowishFritillarybutterfly.webp)

https://gfycat.com/PartialYellowishFritillarybutterfly

![shepard_matzler_predictions_6](https://user-images.githubusercontent.com/15250418/50263627-1953c980-045a-11e9-8924-a7f896f5fc7e.png)
![shepard_matzler_predictions_7](https://user-images.githubusercontent.com/15250418/50263631-21136e00-045a-11e9-87da-0cc2c529c609.png)
![shepard_matzler_predictions_9](https://user-images.githubusercontent.com/15250418/50263644-2c669980-045a-11e9-8574-d887c351f2ad.png)

# Requirements

- Python 3
- Chainer 4+
    `pip3 install chainer`

# Network Architecture

![gqn_conv_draw](https://user-images.githubusercontent.com/15250418/50375239-ad31bb00-063d-11e9-9c1b-151c18dc265d.png)

![gqn_representation](https://user-images.githubusercontent.com/15250418/50375240-adca5180-063d-11e9-8b2a-fb2c3995bc33.png)

# Dataset

https://github.com/musyoku/gqn-dataset-renderer

- **Shepard-Matzler**

![shepard_matzler](https://user-images.githubusercontent.com/15250418/47383748-53496d80-d740-11e8-8db8-e7a25bd1ad5c.gif)

- **Rooms**

![anim](https://user-images.githubusercontent.com/15250418/47347087-7e54a280-d6e9-11e8-93db-47dd2b4efaea.gif)

- **MNIST Dice**

![mnist_dice](https://user-images.githubusercontent.com/15250418/47478271-e4653500-d863-11e8-8d26-1b61cc34cc3b.gif)
