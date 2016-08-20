# eant2

[![Build Status](https://travis-ci.org/pengowen123/eant2.svg?branch=master)](https://travis-ci.org/pengowen123/eant2)

This library is currently broken. I am looking into the problem, which seems to be caused by the CMA-ES crate. Once it is functional it will be released.

An implementation of the [EANT2](http://www.informatik.uni-kiel.de/inf/Sommer/doc/Publications/nts/SiebelSommer-IJHIS2007.pdf) algorithm in Rust. EANT2 is an evolutionary neural network training algorithm. It solves many of the problems associated with evolving topology of neural networks, as well as introducing new concepts. There are already libraries for neural network training, but require a lot of setup to make use of. This library aims to change that. A small number of (completely optional) parameters makes EANT2 easy to use. The goal of this library is to make it easy to include neural networks in any project.

This library has two other parts: [CMA-ES](https://github.com/pengowen123/cmaes) and [CGE][1].

# Usage

Add this to your Cargo.toml:

```
[dependencies]

eant2 = { git = "https://github.com/pengowen123/eant2" }
```

And this to your crate root:

```rust
extern crate eant2;
```

TODO: Add an example here as well as links to more complex examples

After training a neural network, save it to a file:

```rust
network.save_to_file("neural_network.ann").unwrap();
```

To use it elsewhere, use the [CGE library][1]. With it, you can load the network, and evaluate and reset it as many times as you wish. For a freestanding version (for embedded systems), use [cge-core][2].

See the [documentation][3] for complete instructions as well as tips for improving performance of the algorithm. For docmentation on the usage of the neural networks, see the documentation for the CGE library [here](https://pengowen123.github.io/cge/cge/index.html).

# Contributing

If you encounter a bug, please open an issue and I will try to get it fixed. If you have a suggestion, feel free to open an issue or implement it yourself and open a pull request.

[1]: https://github.com/pengowen123/cge
[2]: https://github.com/pengowen123/cge-core
[3]: https://pengowen123.github.io/eant2/eant2/index.html