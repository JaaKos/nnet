#pragma once
#include "matrix.hpp"
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <array>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <numeric>


struct Neuron 
{
    std::vector <double> weights;
    std::vector <double> input;
    double bias;
    double z_value;
    double output;
};

struct Layer
{
    std::vector <Neuron> neurons;
    std::vector <double> mse;
};

struct Filter
{
    Matrix weights;
    Matrix mse;
    double bias;
};

struct conv2dLayer
{
    Matrix input;
    Matrix input_padded;
    Matrix input_no_activation;
    std::vector <Filter> filters;
    std::vector <Matrix> featuremaps;
    std::vector <Matrix> featuremaps_no_activation;
    std::vector <Matrix> d_featuremaps;
    std::vector <Matrix> d_featuremaps_upsample_positions;
};

class net
{
    double learning_rate = 0.001;
    double dropout_rate = 0.2;
    int dense_input_size = 0;
    std::vector <double> getVectorFromLabel(const int label);

public:
    std::vector <conv2dLayer> conv2d_layers;
    std::vector <Layer> layers;
    std::vector <double> first_dense_layer_input;
    std::vector <double> prediction;
    double getLearningRate();
    void setLearningRate(const double lr);

    net(const std::vector <int> & conv2d_filters, const std::vector<int> & layer_sizes, const int input_size);

    void makePrediction(const std::vector<double> & input, const bool dropout = false);

    //double relu(const double x);
    //double d_relu(const double x);
    std::vector<double> softmax(const std::vector<double>& inputs);
    double calculateMeanSquareError(const int label);

    std::vector <double> calculateDeltaMSE(const int label);
    std::vector <double> calculateDeltaReluVector (const Layer & layer);
    void calculateDenseLayersDeltaMSE(const int label);
    void calculateConvFiltersDeltaMSE();
    void adjustBiases();
    void adjustWeights();
    void applyBackPropagation(const int label);

    void saveNetwork(std::string fe_filename, std::string fc_filename);
    void loadNetwork(std::string fe_filename, std::string fc_filename);

    friend double relu(const double x);
    friend double d_relu(const double x);
};