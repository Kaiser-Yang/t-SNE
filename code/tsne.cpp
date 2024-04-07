#include <algorithm>
#include <iostream>
#include <math.h>
#include <ostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <random>

namespace TSNE
{
    using std::vector, std::cin, std::cout, std::ostream, std::max, std::fill,
          std::find_if, std::default_random_engine,std::normal_distribution;
    // n : the number of samples
    // m : the dimension of every sample
    // y ：the result after t-SNE
    // outputDimension : the dimension of result y
    // epoch : the number of iterations,
    //         we'll use max(epoch, 250) as real epoch
    // perp : perplexity, usually chosen in [5, 50], and it must be less than n
    // x : samples, x[i] means the i-th sample
    // g : gradient
    // gain : for adaptive learning rate
    // p : joint probability of samples
    // q : joint probability of result y
    // learningRate : laerning rate
    // lastY : the last y, for momentum gradient descent
    // yMean : the mean number of y, for centralization
    // l, r : the left boundary and the right boundary,
    //        for binary search to get 1 / (2*sigma^2)
    // doubleSigmaSquareInverse : 1 / (2*sigma^2)
    int n, m, outputDimension, epoch;
    double perp;
    constexpr double eps = 1e-7;
    constexpr int EXAGGERATION = 4;
    double learningRate = 100;
    vector<vector<double>> x, p;
    vector<vector<double>> *q;
    vector<vector<double>> y, g, lastY, gain;
    vector<double> yMean;
    vector<double> doubleSigmaSquareInverse, l, r;

    // for generating Gauss Distribution number
    default_random_engine generator;
    normal_distribution<double> d(0., 1e-4);

    // for outputing a matrix
    template<class T>
    ostream & operator<<(ostream & os, const vector<T> &a);
    template<>
    ostream & operator<<(ostream & os, const vector<double> &a);

    // for a matrix divided by a float number
    template<class T>
    void operator/=(vector<T> &a, double b);

    // for a matrix addtion
    template<class T>
    void operator+=(vector<T> &a, const vector<T> &b);

    // this clas is to calculate perplexity of every sample
    class PerpNet
    {
    public:
        // this class is to calculate conditional probability pj|i
        class PLayer
        {
        public:
            vector<vector<double>> output;
            const vector<double> *doubleSigmaSquareInverse;
            vector<vector<double>> disSquare;

            void setDoubleSigmaSquareInverse(const vector<double>& value)
            {
                this->doubleSigmaSquareInverse = &value;
            }

            vector<vector<double>> & operator()(vector<vector<double>> x)
            {
                // we only calculate norm(x[i] - x[j])^2 once
                if (disSquare.empty()) {
                    disSquare.resize(x.size(), vector<double>(x.size()));
                    double normSquare = 0;
                    for (int i = 0; i < x.size(); i++) {
                        for (int j = i + 1; j < x.size(); j++) {
                            normSquare = 0;
                            assert(x[i].size() == x[j].size());
                            for (int k = 0; k < x[i].size(); k++) {
                                normSquare += pow(x[i][k] - x[j][k], 2);
                            }
                            disSquare[i][j] = disSquare[j][i] = normSquare;
                        }
                    }
                }
                if (output.empty()) {
                    output.resize(x.size(), vector<double>(x.size()));
                }
                // calculate pj|i, output[i][j] is pj|i
                double piSum = 0;
                for (int i = 0; i < x.size(); i++) {
                    piSum = 0;
                    for (int j = 0; j < x.size(); j++) {
                        if (j == i) { continue; }
                        output[i][j] = exp(-disSquare[i][j] *
                                           (*doubleSigmaSquareInverse)[i]);
                        piSum += output[i][j];
                    }
                    assert(piSum > eps);
                    for (int j = 0; j < x.size(); j++) {
                        output[i][j] /= piSum;
                    }
                }
                return output;
            }

            void init()
            {
                output.clear();
                doubleSigmaSquareInverse = nullptr;
                disSquare.clear();
            }
        } p;
        
        // this class is for calculating H
        class HLayer
        {
        public:
            vector<double> output;

            vector<double> & operator()(const vector<vector<double>> &x)
            {
                if (output.empty()) { output.resize(x.size()); }
                double h = 0;
                for (int i = 0; i < x.size(); i++) {
                    h = 0;
                    for (int j = 0; j < x.size(); j++) {
                        if (x[i][j] > eps) { h -= x[i][j] * log2(x[i][j]); }
                    }
                    output[i] = h;
                }
                return output;
            }

            void init() { output.clear(); }
        } h;

        // this class is for calculating Perp
        class PerpLayer
        {
        public:
            vector<double> output;

            vector<double> & operator()(const vector<double> x)
            {
                if (output.empty()) { output.resize(x.size()); }
                for (int i = 0; i < x.size(); i++) { output[i] = pow(2, x[i]); }
                return output;
            }

            void init() { output.clear(); }
        } perp;

        void setDoubleSigmaSquareInverse(const vector<double> &value)
        {
            p.setDoubleSigmaSquareInverse(value);
        }

        vector<double> & operator()(const vector<vector<double>> &x)
        {
            return perp(h(p(x)));
        }

        void init()
        {
            perp.init();
            h.init();
            p.init();
        }
    } perpNet;

    // this clasl is to calculate joint probability q
    class QLayer
    {
    public:
        vector<vector<double>> output;
        vector<vector<double>> disSquarePlusInverse;

        vector<vector<double>> & operator()(const vector<vector<double>> &x)
        {
            // disSquarePlusInverse stores 1 / (1 + ||xi - xj||^2)
            // this will be reused when calculating gradient
            if (disSquarePlusInverse.empty()) {
                disSquarePlusInverse.resize(x.size(), vector<double>(x.size()));
            }
            if (output.empty()) {
                output.resize(x.size(), vector<double>(x.size()));
            }
            double qSum = 0;
            double normSquare = 0;
            for (int i = 0; i < x.size(); i++) {
                for (int j = i + 1; j < x.size(); j++) {
                    normSquare = 0;
                    assert(x[i].size() == x[j].size());
                    for (int k = 0; k < x[i].size(); k++) {
                        normSquare += pow(x[i][k] - x[j][k], 2);
                    }
                    output[i][j] = output[j][i]
                                 = disSquarePlusInverse[i][j]
                                 = disSquarePlusInverse[j][i]
                                 = 1 / (1 + normSquare);
                    qSum += output[i][j] * 2;
                }
            }
            assert(qSum > eps);
            for (int i = 0; i < x.size(); i++) {
                for (int j = i + 1; j < x.size(); j++) {
                    output[i][j] = output[j][i] = max(output[i][j] / qSum,
                                                      1e-100);
                }
            }
            return output;
        }

        void init() {
            output.clear();
            disSquarePlusInverse.clear();
        }
    } qLayer;

    // binary search to get 1 / (2*sigma^2)
    void getDoubleSigmaSquareInverse()
    {
        perpNet.setDoubleSigmaSquareInverse(r);
        while (true) {
            auto &&perp = perpNet(x);
            auto &&it = find_if(perp.begin(), perp.end(), [](double item) {
                            return item > TSNE::perp;
                        });
            if (it == perp.end()) { break; }
            r += r;
        }
        perpNet.setDoubleSigmaSquareInverse(doubleSigmaSquareInverse);
        assert(r.size() > 0 && l.size() > 0);
        while (r[0] - l[0] >= eps) {
            for (int i = 0; i < n; i++) {
                doubleSigmaSquareInverse[i] = (l[i] + r[i]) / 2;
            }
            // get Perp with current 1 / (2*sigma^2)
            auto &&perp = perpNet(x);
            for (int i = 0; i < n; i++) {
                if (perp[i] < TSNE::perp) {
                    r[i] = doubleSigmaSquareInverse[i];
                }
                else { l[i] = doubleSigmaSquareInverse[i]; }
            }
        }
    }

    // calculate gredient
    void getGradient()
    {
        double scale = 0;
        for (int i = 0; i < n; i++) {
            fill(g[i].begin(), g[i].end(), 0);
            for (int j = 0; j < n; j++) {
                // this scale is to prevent re-calculation
                scale = 4 * (p[i][j] - (*q)[i][j]) *
                        qLayer.disSquarePlusInverse[i][j];
                for (int d = 0; d < outputDimension; d++) {
                    g[i][d] += scale * (y[i][d] - y[j][d]);
                }
            }
        }
    }

    // get the signature of input x
    int sign(double x) { return x > 0 ? 1 : (x < 0 ? -1 : 0); }

    // momentum gradient descent epoch times with momentum
    // this will update learning rate adaptively
    void gradientDescent(int epoch, double momentum)
    {
        double step = 0;
        for (int t = 0; t < epoch; t++) {
            // calculate joint probability of current result y
            q = &qLayer(y);
            getGradient();
            fill(yMean.begin(), yMean.end(), 0);
            for (int i = 0; i < n; i++) {
                for (int d = 0; d < outputDimension; d++) {
                    // update learning rate adaptively
                    // we don't update learningRate variable
                    // we use learningRate * gain[i][d] as the new learning rate
                    gain[i][d] = max(sign(g[i][d]) == sign(lastY[i][d]) ?
                                 gain[i][d] * 0.8 :
                                 gain[i][d] + 0.2, 0.01);
                    // get step
                    step = momentum * (y[i][d] - lastY[i][d]) -
                           learningRate * gain[i][d] * g[i][d];
                    lastY[i][d] = y[i][d];
                    y[i][d] += step;
                    // for centralization
                    yMean[d] += y[i][d];
                }
            }
            for (int d = 0; d < outputDimension; d++) { yMean[d] /= n; }
            // centralization
            for (int i = 0; i < n; i++) {
                for (int d = 0; d < outputDimension; d++) {
                    y[i][d] -= yMean[d];
                }
            }
            // output y after every iteration
            cout << y;
        }
    }

    void init()
    {
        assert(perp < n);
        assert(n > 0);
        assert(outputDimension > 0);
        assert(outputDimension < m);
        perpNet.init();

        p.clear();
        p.resize(n, vector<double>(n));

        q = nullptr;

        y.clear();
        y.resize(n, vector<double>(outputDimension));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < outputDimension; j++) {
                y[i][j] = d(generator);
            }
        }

        lastY = y;

        doubleSigmaSquareInverse.clear();
        doubleSigmaSquareInverse.resize(n);

        l.clear();
        l.resize(n, 0);

        r.clear();
        r.resize(n, eps);
        
        g.clear();
        g.resize(n, vector<double>(n));

        yMean.clear();
        yMean.resize(outputDimension);

        learningRate = 100;

        gain.clear();
        gain.resize(n, vector<double>(outputDimension, 1.));

        epoch = max(epoch, 250);
    }

    // run t-SNE
    void run()
    {
        // initialize matrixs and generate initial y
        init();
        // compute 1 / (2 sigma^2)
        getDoubleSigmaSquareInverse();
        // compute joint probability pij
        auto &&output = perpNet.p(x);
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                p[i][j] = p[j][i] =
                          max(1e-100, (output[i][j] + output[j][i]) / (2 * n)) *
                          EXAGGERATION;
            }
        }
        // gradient descent with epoch and momentum scale
        gradientDescent(50, 0.5);
        p /= EXAGGERATION;
        gradientDescent(200, 0.5);
        gradientDescent(epoch - 250, 0.8);
    }

    template<class T>
    void operator+=(vector<T> &a, const vector<T> &b)
    {
        assert(a.size() == b.size());
        for (int i = 0; i < b.size(); i++) { a[i] += b[i]; }
    }

    template<class T>
    ostream & operator<<(ostream & os, const vector<T> &a)
    {
        for (int i = 0; i < a.size(); i++) { os << a[i] << "\n"; }
        return os;
    }

    template<>
    ostream & operator<<(ostream & os, const vector<double> &a)
    {
        for (int i = 0; i < a.size(); i++) { os << a[i] << " "; }
        return os;
    }

    template<class T>
    void operator/=(vector<T> &a, double b)
    {
        assert(b != 0);
        for (int i = 0; i < a.size(); i++) { a[i] /= b; }
    }
};


int main()
{
    std::ios::sync_with_stdio(false);
    using namespace TSNE;
    // get data from data/data.in and output data to data/data.out using freopen
    auto in = freopen("data/data.in", "r", stdin);
    auto out = freopen("data/data.out", "w", stdout);
    assert(in != nullptr);
    assert(out != nullptr);
    cin >> n >> m >> outputDimension >> perp >> epoch;
    x.resize(n, vector<double>(m));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            cin >> x[i][j];
        }
    }
    // run t-SNE
    run();
    // this is neccesary
    // because fclose will not flush the data that in the buffer of cout
    cout.flush();
    fclose(in);
    fclose(out);
    return 0;
}