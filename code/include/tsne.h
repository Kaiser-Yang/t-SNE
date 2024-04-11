#ifndef TSNE_H

#define TSNE_H

#include <vector>
#include <string>

namespace TSNE
{
    // These variables below are must be set, before t-SNE run:
    // n: the number of samples,
    //    when enableRandomWalk is true, this means the number of landmarks
    // m: the dimension of every sample
    // outputDimension: the dimension of result y
    // epoch: the number of iterations,
    //        we'll use max(epoch, 250) as real epoch,
    //        which means default number is 250
    // totalSampleNum: this means the number of all samples,
    //                 when enableRandomWalk is 0, this should be same with n
    // perp: perplexity, usually chosen in [5, 50], and it must be less than n
    // learningRate: laerning rate, default value is 100
    // enableRandomWalk: whether or not use random walk, the default value is 0
    // neighborNum: this variable is to find the neighborNum nearest neighbor,
    //              we will use min(round(perp), n-1) as neighborNum
    // randomWalkTimes: for every landmark, we random walk randomWalkTimes times
    //                  to use the frequency as the probability,
    //                  the default value is 10000
    // randomWalkTimesEveryTurn: the max random walk times every random walk
    //                           turn, the default value is n, if it is set,
    //                           we'll use max(randomWalkTimesEveryTurn, n)

    // The output of t-SNE run:
    // y ：the result after t-SNE

    // These variables below are configurable:
    // eps: precision for binary search, the default value is 1e-7
    extern int n, m, outputDimension, epoch;
    extern std::vector<std::vector<double>> totalSample;
    extern double perp;
    extern int exaggeration;
    extern double learningRate;
    extern bool enableRandomWalk;
    extern int totalSampleNum, neighborNum;
    extern int randomWalkTimes;
    extern int randomWalkTimesEveryTurn;
    extern std::vector<std::vector<double>> y;
    extern double eps;

    // after setting all the variables that required, 
    // call TSNE::run() to start t-SNE
    // outputFilename: the filename where y should be outputted
    extern void run(const std::string &outputFilename = "data/data.out");
}

#endif
