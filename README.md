# INTRODUCTION

Implementation of [t-SNE](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) with `C++`.

Visualization of the `t-SNE` process with `Python`.

# ENVIRONMENT LIST

* `cmake 3.0.0`  or later;
* `python 3.8.0` or later;
* `c++ compiler`  must support `C++17`;
* `numpy, imageio, matplotlib, torch, sklearn` for your `python` is required;

**TIPS: if you only build `tsne` manually, you just need a `c++ compiler` supporting `C++17` and use the command `g++ tsne.cpp -o tsne --std=c++17 -O3 -DNDEBUG` (if your compiler is `g++`)  to get the executable.**

# NOTEs

* If you build `tsne` without `cmake`, you can use `g++ tsne.cpp -o tsne --std=c++17 -O3 -DNDEBUG` command;
* If your platform is `UNIK-like`, you can use `bash start.sh` to build and run this code automatically, otherwise, you can execute commands as the `start.sh` does;
* Make sure your working directory of every script and `tsne` is `code/` to make finding data with relative paths possible;
* The execution of `tsne` takes a long duration, so you are supposed to wait patiently, or you can update all `n`s, which means the number of samples, in `*.py` to run on less data, the default `n` is `6000`;

# IMPLEMENTATION OPTIMIZATIONs

* There are many symmetric calculation in formulas with i and j, so we just need calculate half elements of a matrix, such as the code calculating joint probability $p_{ij}$ of the input data:

  ```c++
  for (int i = 0; i < n; i++) {
      for (int j = i + 1; j < n; j++) {
          p[i][j] = p[j][i] = max(1e-100, (output[i][j] + output[j][i]) / (2 * n)) * EXAGGERATION;
      }
  }
  ```

* Try not to copy from a matrix, such as the code of method `operator+=`, which update elements of a matrix directly:

  ```c++
  template<class T>
  void operator+=(vector<T> &a, const vector<T> &b)
  {
      for (int i = 0; i < b.size(); i++) { a[i] += b[i]; }
  }
  ```

* Where is a $\sigma_i$, there is a $\frac{1}{2\sigma_i^2}$. Therefore, we just need calculate $\frac{1}{2\sigma_i^2}$ rather than $\sigma_i$, such as the code calculating the conditional probability $p_{j\vert i}$ of input data ( `(*doubleSigmaSquareInverse)[i]`means $\frac{1}{2\sigma_i^2}$ exactly):

  ```c++
  double piSum = 0;
  for (int i = 0; i < x.size(); i++) {
      piSum = 0;
      for (int j = 0; j < x.size(); j++) {
          if (j == i) { continue; }
          output[i][j] = exp(-disSquare[i][j] * (*doubleSigmaSquareInverse)[i]);
          piSum += output[i][j];
      }
      for (int j = 0; j < x.size(); j++) { output[i][j] /= piSum; }
  }
  ```

* There are many results that will be used more than one time, such as the `L2` of input data. Therefore we can calculate once and restore them for future use, such as the code calculating `L2` of input data:

  ```c++
  if (disSquare.empty()) {
      disSquare.resize(x.size(), vector<double>(x.size()));
      double normSquare = 0;
      for (int i = 0; i < x.size(); i++) {
          for (int j = i + 1; j < x.size(); j++) {
              normSquare = 0;
              assert(x[i].size() == x[j].size());
              for (int k = 0; k < x[i].size(); k++) { normSquare += pow(x[i][k] - x[j][k], 2); }
              disSquare[i][j] = disSquare[j][i] = normSquare;
          }
      }
  }
  ```

# POSSIBLE IMPROVEMENT

* Using multi-thread to improve is obvious.

# RESULT

The result of one random experiment (this figure is a `gif` file, which makes it possible to restart by saving it or opening it in a new tab) :

![result](./result.gif)