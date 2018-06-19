# :construction: Work in Progress :construction:

# Neural scene representation and rendering

[https://deepmind.com/blog/neural-scene-representation-and-rendering/](https://deepmind.com/blog/neural-scene-representation-and-rendering/)

**Todo**

- [x] レンダラーの実装 / Simulator
- [ ] ニューラルネットの実装 / Neural network
- [ ] 学習 / Training

# Requirements

- pybind11
- Python 3
- OpenGL
- Ubuntu
- C++14

# Installation

**pybind11**

```
pip3 install pybind11 --user
```

**Renderer**

```
cd three
make
```

**Viewer**

```
cd viewer
make
```

# Dataset

## Rooms with multiple objects

## Shepard-Metzler objects

![Dataset](https://qiita-image-store.s3.amazonaws.com/0/109322/a2777cad-36b6-4c28-b8e8-a6bca16af03a.gif)

![Objects](https://qiita-image-store.s3.amazonaws.com/0/109322/ca32401b-ad69-cbc0-572b-8f9b8e0b62b4.gif)

![Environment](https://qiita-image-store.s3.amazonaws.com/0/109322/bf87c553-02ce-9568-0cc5-32b2d704d24b.gif)


# Training