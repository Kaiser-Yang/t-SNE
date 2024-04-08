# INTRODUCTION

Implementation of [t-SNE](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) with `C++` (including the random walk version).

Visualization of the `t-SNE` process with `Python`.

# ENVIRONMENT LIST

* `cmake 3.8.0`  or later;
* `python 3.8.0` or later;
* `c++ compiler`  must support `C++17`;
* `numpy, imageio, matplotlib, torch, sklearn` for your `python` is required;

**TIPS: if you only build `tsne` manually, you just need a `c++ compiler` supporting `C++17` and use the command `g++ tsne.cpp -o tsne --std=c++17 -O3 -DNDEBUG` (if your compiler is `g++`)  to get the executable.**

# NOTEs

* If you build `tsne` without `cmake`, you can use `g++ tsne.cpp -o tsne --std=c++17 -O3 -DNDEBUG` command;
* If your platform is `UNIK-like`, you can use `bash start.sh` to build and run this code automatically, otherwise, you can execute commands as the `start.sh` does;
* Make sure your working directory of every script and `tsne` is `code/` to make finding data with relative paths possible;
* The execution of `tsne` takes a long duration, so you are supposed to wait patiently, or you can update all `n`s, which means the number of samples, in `*.py` to run on less data, the default `n` is `6000`;
* I've added arguements for `start.sh`, and those are `n`, `epoch`, `enableRandomWalk` and `totalSampleNum` in order.
  * For example, `bash start.sh 6000 1000` means `6,000` samples and `1,000` iterations without random walk, which is default (same with `bash start.sh`); `bash start.sh 6000 1000 1 60000` means `6,000` samples will be showed (if you don't understrand this, you are supposed to read the random walk part of [t-SNE](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)), `1,000` iterations, enable random walk and the number of total samples is `6,0000`
  * If `enableRandomWalk` is `0` (the default value), you should not set `totalSampleNum` be not equal with `n`.
* There is a typo in the pseudo code of [the origin paper](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf). Of the gradient descent part, there should be a minus rather than a plus, which should be $y_i^{t+1} = y_i^t - \eta \frac{\delta C}{\delta y_i} + \alpha(t) (y_i^t - y_i^{t - 1})$;
* There are some output information when running t-SNE. The train process are seperated to `3` parts (`50` iterations, `200` iterations and `epoch - 250` iterations), so the train process rates are for every part in sequence.

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

* In random walk, we limit the times of each turn to avoid random walk on a small circle for a long time:
```c++
if (nearestLandMark[start] == UNKNOWN) {
    nearestLandMark[start] = getNearestLandMark(selectedID[start]);
}
if (nearestLandMark[start] == UNREACHABLE) { return UNREACHABLE; }
int now = selectedID[start];
int randomWalkTimes = randomWalkTimesEveryTurn;
while (randomWalkTimes-- &&
       (now == selectedID[start] || selectedIDMap.count(now) == 0)) {
    now = randomWalkGraph[now][distribution[now](generator)].to;
}
if (selectedIDMap.count(now) == 0) {
    return selectedIDMap[nearestLandMark[start]];
}
return selectedIDMap[now];
```

# POSSIBLE IMPROVEMENT

* Using multi-thread to improve is obvious.

# Future Work
* ~~Implementing t-SNE with random walk to train on bigger data sets.~~(this has been implemented!)

# RESULT

The result of one random experiment without random walk(training on `6,000` MNIST images for `1,000` iteration, this figure is a `gif` file, which makes it possible to restart by saving it or opening it in a new tab) :

![result](./result.gif)

The result of one random experiment (training on `6,000` MNIST images for `1,000` iteration, but using all `60,000` MNIST images with random walk, this figure is a `gif` file, which makes it possible to restart by saving it or opening it in a new tab):
![result_random_walk](./result_random_walk.gif)
