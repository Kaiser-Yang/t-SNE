#include <algorithm>
#include <ctime>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <ostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cassert>
#include <cmath>
#include <random>
#include <queue>

#include "tsne.h"

namespace TSNE
{
    // n: the number of samples,
    //    when enableRandomWalk is true, this means the number of landmarks
    // m: the dimension of every sample
    // y ：the result after t-SNE
    // outputDimension: the dimension of result y
    // epoch: the number of iterations,
    //        we'll use max(epoch, 250) as real epoch
    // perp: perplexity, usually chosen in [5, 50], and it must be less than n
    // x: the actual trained samples, x[i] means the i-th sample,
    //    when enableRandomWalk is false, x is totalSample,
    //    when enableRandomWalk is true, x is randomly selected from totalSample
    // g: gradient
    // gain: for adaptive learning rate
    // p: joint probability of samples
    // q: joint probability of result y
    // learningRate: laerning rate
    // lastY: the last y, for momentum gradient descent
    // yMean: the mean number of y, for centralization
    // l, r: the left boundary and the right boundary,
    //        for binary search to get 1 / (2*sigma^2)
    // doubleSigmaSquareInverse: 1 / (2*sigma^2)
    // enableRandomWalk: whether or not use random walk
    // totalSampleNum: this means the number of all samples.
    //                 when enableRandomWalk is 0, this should be same with n
    // neighborNum: this variable is to find the neighborNum nearest neighbor,
    //              we will use min(round(perp), n-1) as neighborNum
    // randomWalkTimes: for every landmark, we random walk randomWalkTimes times
    //                  to use the frequency as the probability
    // randomWalkTimesEveryTurn: the max random walk times every random walk
    //                           turn, the default value is n, if it is set,
    //                           we'll use max(randomWalkTimesEveryTurn, n)
    // disSquare: disSquare[i][j] means ||totalSample[i] - totalSample[j]||^2
    // selectedID: selectedID[i] means the i-th landmark position in totalSample
    // selectedIDMap: selectedIDMap[selectedID[i]] = i
    // randomWalkGraph: store the graph for random walk
    using std::vector, std::ostream;
    int n, m, outputDimension, epoch;
    double perp;
    double eps = 1e-7;
    int exaggeration = 4;
    double learningRate = 100;
    vector<vector<double>> x;
    vector<vector<double>> totalSample;
    vector<vector<double>> disSquare;
    vector<vector<double>> p, *q;
    vector<vector<double>> y, g, lastY, gain;
    vector<double> yMean;
    vector<double> doubleSigmaSquareInverse, l, r;
    bool enableRandomWalk;
    int totalSampleNum, neighborNum;
    int randomWalkTimes = 1e4;
    int randomWalkTimesEveryTurn;
    vector<int> selectedID;
    std::unordered_map<int, int> selectedIDMap;

    struct RandomWalkEdge { double w; int to; };
    vector<vector<RandomWalkEdge>> randomWalkGraph;

    // for generating Gauss Distribution number
    std::default_random_engine generator;
    std::normal_distribution<double> d(0., 1e-4);

    // output y to file
    std::ofstream fileOutstream;

    // landmark state, nearestLandMark[id] means id's nearest landmark
    enum LANDMARK_STATE
    {
        UNKNOWN = -2,
        UNREACHABLE = -1
    };
    vector<int> nearestLandMark;

    // for outputing a matrix
    template<class T>
    ostream & operator<<(ostream & os, const vector<T> &a);
    template<>
    ostream & operator<<(ostream & os, const vector<double> &a);
    template<>
    ostream & operator<<(ostream & os, const vector<int> &a);

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
            // vector<vector<double>> disSquare;

            void setDoubleSigmaSquareInverse(const vector<double>& value)
            {
                this->doubleSigmaSquareInverse = &value;
            }

            vector<vector<double>> & operator()(const vector<vector<double>> &x)
            {
                if (output.empty()) {
                    output.resize(x.size(), vector<double>(x.size()));
                }
                // calculate pj|i, output[i][j] is pj|i
                double piSum = 0;
                for (int i = 0; i < x.size(); i++) {
                    piSum = 0;
                    for (int j = 0; j < x.size(); j++) {
                        if (j == i) { continue; }
                        output[i][j] = exp(-disSquare[selectedID[i]][selectedID[j]] *
                                           (*doubleSigmaSquareInverse)[i]);
                        piSum += output[i][j];
                    }
                    assert(piSum > 1e-100);
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
                // disSquare.clear();
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
                        if (x[i][j] > 1e-100) { h -= x[i][j] * log2(x[i][j]); }
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

            vector<double> & operator()(const vector<double> &x)
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
            assert(qSum > 1e-100);
            for (int i = 0; i < x.size(); i++) {
                for (int j = i + 1; j < x.size(); j++) {
                    output[i][j] = output[j][i] = std::max(output[i][j] / qSum,
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
        clock_t startTime = clock();
        perpNet.setDoubleSigmaSquareInverse(r);
        while (true) {
            auto &&perp = perpNet(x);
            auto &&it = std::find_if(perp.begin(), perp.end(), [](double item) {
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
        clock_t endTime = clock();
        std::cout << "time of getDoubleSigmaSquareInverse: "
                  << (double)(endTime - startTime) / CLOCKS_PER_SEC
                  << " (s)"
                  << std::endl;
    }

    // calculate gredient
    void getGradient()
    {
        double scale = 0;
        for (int i = 0; i < n; i++) {
            std::fill(g[i].begin(), g[i].end(), 0);
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
            std::fill(yMean.begin(), yMean.end(), 0);
            for (int i = 0; i < n; i++) {
                for (int d = 0; d < outputDimension; d++) {
                    // update learning rate adaptively
                    // we don't update learningRate variable
                    // we use learningRate * gain[i][d] as the new learning rate
                    gain[i][d] = std::max(sign(g[i][d]) == sign(lastY[i][d]) ?
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
            fileOutstream << y << '\n';
            if (epoch <= 100 && t % 10 == 0 ||
                (epoch > 100 && t % 50 == 0) ||
                t == epoch - 1) {
                std::cout << "train process: "
                          << (t + 1) * 100. / epoch
                          << "%" << std::endl;
            }
        }
    }

    // select n samples from totalSample randomly.
    void randomSelection()
    {
        clock_t startTime = clock();
        std::unordered_set<int> idx;
        for (int i = 0; i < totalSampleNum; i++) { idx.insert(i); }
        std::sample(idx.begin(), idx.end(), std::back_inserter(selectedID),
                    n, std::mt19937{ std::random_device{}() });
        sort(selectedID.begin(), selectedID.end());
        // this is to avoid copy from totalSample
        // we would swap again when the train finished.
        for (int i = 0; i < selectedID.size(); i++) {
            x[i].swap(totalSample[selectedID[i]]);
            selectedIDMap[selectedID[i]] = i;
        }
        fileOutstream << selectedID << '\n';
        clock_t endTime = clock();
        std::cout << "time of randomSelection: "
                  << (double)(endTime - startTime) / CLOCKS_PER_SEC
                  << " (s)"
                  << std::endl;
    }

    void init(const std::string &outputFilename)
    {
        assert(perp < n);
        assert(n > 0);
        assert(outputDimension > 0);
        assert(outputDimension < m);
        assert(enableRandomWalk || n == totalSampleNum);

        fileOutstream = std::ofstream(outputFilename);

        if (enableRandomWalk) {
            neighborNum = std::min((int)round(perp), totalSampleNum - 1);
            randomWalkGraph.resize(totalSampleNum);
            randomWalkTimesEveryTurn = std::max(randomWalkTimesEveryTurn, n);
            nearestLandMark.resize(n, UNKNOWN);
        } else {
            perpNet.init();
            doubleSigmaSquareInverse.clear();
            doubleSigmaSquareInverse.resize(n);

            l.clear();
            l.resize(n, 0);

            r.clear();
            r.resize(n, eps);
        }

        p.clear();
        p.resize(n, vector<double>(n));

        q = nullptr;

        x.resize(n, vector<double>(m));

        y.clear();
        y.resize(n, vector<double>(outputDimension));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < outputDimension; j++) {
                y[i][j] = d(generator);
            }
        }

        lastY = y;
        
        g.clear();
        g.resize(n, vector<double>(n));

        yMean.clear();
        yMean.resize(outputDimension);

        learningRate = 100;

        gain.clear();
        gain.resize(n, vector<double>(outputDimension, 1.));

        epoch = std::max(epoch, 250);

        disSquare.resize(totalSampleNum, vector<double>(totalSampleNum));
    }

    // store (input[i] - input[j])^2 in res[i][j]
    void getDisSquare(const vector<vector<double>> &input,
                      vector<vector<double>> &res)
    {
        clock_t startTime = clock();
        assert(res.size() == input.size());
        double normSquare = 0;
        for (int i = 0; i < input.size(); i++) {
            assert(res[i].size() == input.size());
            for (int j = i + 1; j < input.size(); j++) {
                normSquare = 0;
                assert(input[i].size() == input[j].size());
                for (int k = 0; k < input[i].size(); k++) {
                    normSquare += pow(input[i][k] - input[j][k], 2);
                }
                res[i][j] = res[j][i] = normSquare;
            }
        }
        clock_t endTime = clock();
        std::cout << "time of getDisSquare: "
                  << (double)(endTime - startTime) / CLOCKS_PER_SEC
                  << " (s)"
                  << std::endl;
    }

    // create random walk graph edges whose start point is u
    void getKNearestNeighbor(int u)
    {
        std::priority_queue<std::pair<double, int>> maxHeap;
        for (int v = 0; v < totalSampleNum; v++) {
            if (u == v) { continue; }
            if (maxHeap.size() < neighborNum) {
                maxHeap.push({disSquare[u][v], v});
            } else if (maxHeap.top().first > disSquare[u][v]) {
                maxHeap.pop();
                maxHeap.push({disSquare[u][v], v});
            }
        }
        assert(maxHeap.size() == neighborNum);
        while (!maxHeap.empty()) {
            randomWalkGraph[u].push_back({exp(-maxHeap.top().first),
                                          maxHeap.top().second});
            maxHeap.pop();
        }
    }

    // get the nearest landmark of start that different from start
    // this graph is sparse, so we use dijkstra
    int getNearestLandMark(int start)
    {
        using std::priority_queue, std::pair, std::greater, std::numeric_limits;
        priority_queue<pair<double, int>,
                       vector<pair<double, int>>,
                       greater<pair<double, int>>> q;
        vector<bool> used(totalSampleNum, false);
        vector<double> dis(totalSampleNum, numeric_limits<double>::max() / 10);
        dis[start] = 0;
        q.push({dis[start], start});
        while (!q.empty()) {
            auto item = q.top();
            q.pop();
            int u = item.second;
            if (used[u]) { continue; }
            if (selectedIDMap.count(u) > 0) { return u; }
            for (auto && edge : randomWalkGraph[u]) {
                int v = edge.to;
                if (dis[v] > dis[u] + disSquare[u][v]) {
                    dis[v] = dis[u] + disSquare[u][v];
                    q.push({dis[v], v});
                }
            }
        }
        return UNREACHABLE;
    }

    // one turn of random walk,
    // this will stop when reaching another landmark,
    // return the reached landmark index in x
    // maker sure dis(gen) will be in [0, neighborNum)
    // if we don't limit the random walk times, 
    // it will be super time consuming,
    // so every turn of random walk, we walk most randomWalkTimesEveryTurn times,
    // if we didn't reach another landmark in randomWalkTimesEveryTurn times,
    // return the nearest landmark from start
    // return -1 if there is no reachable landmark that different from start
    template<class Generator, class Distribution>
    int randomWalk(int start, Generator &generator, vector<Distribution> &distribution)
    {
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
    }

    // calculate joint probability p with random walk
    void randomWalkP()
    {
        clock_t startTime = clock();
        assert(neighborNum != 0);
        for (int u = 0; u < totalSampleNum; u++) { getKNearestNeighbor(u); }
        clock_t endTime = clock();
        std::cout << "time of getKNearestNeighbor for all nodes: "
                  << (double)(endTime - startTime) / CLOCKS_PER_SEC
                  << " (s)"
                  << std::endl;
        std::random_device randomDevice;
        std::mt19937 generator(randomDevice());
        vector<std::discrete_distribution<int>> distribution;
        vector<double> w(neighborNum);
        for (int i = 0; i < totalSampleNum; i++) {
            assert(w.size() == randomWalkGraph[i].size());
            for (int j = 0; j < randomWalkGraph[i].size(); j++) {
                w[j] = randomWalkGraph[i][j].w;
            }
            distribution.push_back(std::discrete_distribution(w.begin(),
                                                              w.end()));
        }
        vector<int> cnt(n);
        int ret = 0;
        for (int i = 0; i < n; i++) {
            std::fill(cnt.begin(), cnt.end(), 0);
            for (int j = 0; j < randomWalkTimes; j++) {
                ret = randomWalk(i, generator, distribution);
                if (ret == UNREACHABLE) { break; }
                cnt[ret]++;
            }
            if (ret != UNREACHABLE) {
                for (int j = 0; j < n; j++) {
                    p[i][j] = (double)cnt[j] / randomWalkTimes;
                }
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                p[i][j] = p[j][i] = std::max(1e-100,
                                             (p[i][j] + p[j][i]) / (2 * n)) *
                                             exaggeration;
            }
        }
        endTime = clock();
        std::cout << "time of calculating joint probability with random walk: "
                  << (double)(endTime - startTime) / CLOCKS_PER_SEC
                  << " (s)"
                  << std::endl;
    }

    // calculate joint probability p using the normal method
    void normalP()
    {
        clock_t startTime = clock();
        // compute 1 / (2 sigma^2)
        getDoubleSigmaSquareInverse();
        // compute joint probability pij
        auto &&output = perpNet.p(x);
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                p[i][j] = p[j][i] = std::max(1e-100,
                                             (output[i][j] + output[j][i]) / (2 * n)) *
                                             exaggeration;
            }
        }
        clock_t endTime = clock();
        std::cout << "time of calculating joint probability: "
                  << (double)(endTime - startTime) / CLOCKS_PER_SEC
                  << " (s)"
                  << std::endl;
    }

    // run t-SNE
    void run(const std::string &outputFilename)
    {
        clock_t startTime = clock();
        // initialize matrixs and generate initial y
        init(outputFilename);
        getDisSquare(totalSample, disSquare);
        // select n samples from totalSample randomly.
        randomSelection();
        if (enableRandomWalk) { randomWalkP(); }
        else { normalP(); }
        // gradient descent with epoch and momentum scale
        gradientDescent(50, 0.5);
        p /= exaggeration;
        gradientDescent(200, 0.5);
        gradientDescent(epoch - 250, 0.8);
        // swap again to make totalSample be original
        for (int i = 0; i < selectedID.size(); i++) {
            x[i].swap(totalSample[selectedID[i]]);
        }
        fileOutstream.close();
        clock_t endTime = clock();
        std::cout << "time of run t-SNE: "
                  << (double)(endTime - startTime) / CLOCKS_PER_SEC
                  << " (s)"
                  << std::endl;
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
        for (int i = 0; i < a.size(); i++) {
            os << a[i] << (i == a.size() - 1 ? "" : "\n");
        }
        return os;
    }

    template<>
    ostream & operator<<(ostream & os, const vector<double> &a)
    {
        for (int i = 0; i < a.size(); i++) {
            os << a[i] << (i == a.size() - 1 ? "" : " ");
        }
        return os;
    }

    template<>
    ostream & operator<<(ostream & os, const vector<int> &a)
    {
        for (int i = 0; i < a.size(); i++) {
            os << a[i] << (i == a.size() - 1 ? "" : " ");
        }
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
    using std::cin;
    // get data from data/data.in and output data to data/data.out using freopen
    auto in = freopen("data/data.in", "r", stdin);
    assert(in != nullptr);
    cin >> n >> m >> outputDimension >> perp >> epoch
        >> enableRandomWalk >> totalSampleNum;
    totalSample.resize(totalSampleNum, vector<double>(m));
    for (int i = 0; i < totalSampleNum; i++) {
        for (int j = 0; j < m; j++) {
            cin >> totalSample[i][j];
        }
    }
    // run t-SNE
    run();
    fclose(in);
    return 0;
}
