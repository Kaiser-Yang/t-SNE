# INTRODUCTION

Implementation of [t-SNE](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) with `C++`.

Visualization of the `t-SNE` process with `Python`.

# Environment List

* `cmake 3.0.0`  or later;
* `python 3.8.0` or later;
* `c++ compiler`  must support `C++17`;
* `numpy, imageio, matplotlib, torch, sklearn` for your `python` is required;

**TIPS: if you only build `tsne` manually, you just need a `c++ compiler` supporting `C++17` and use the command `g++ tsne.cpp -o tsne --std=c++17 -O3 -DNDEBUG` (if your compiler is `g++`)  to get the executable.**

# NOTE

* If you build with out `cmake`, you can use `g++ tsne.cpp -o tsne --std=c++17 -O3 -DNDEBUG`;
* If your platform is `UNIK-like`, you can use `bash start.sh` run this code automatically, otherwise, you can run as the `start.sh` does;
* Make sure your working directory is `code/` to make finding data with relative paths possible;
* The execution of `tsne` takes a long duration, so you should wait patiently, or you can update all `n`s, which means the number of samples, in `.py` to run on less data, the default `n` is `6000`;

# POSSIBLE IMPROVEMENT

* Using multi-thread to improve is obvious.

# RESULT

The result of one random experiment:

![result](./result.gif)