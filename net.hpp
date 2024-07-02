#pragma once
#include "matrix.hpp"
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <array>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <algorithm>


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
    std::vector <double> label_to_vector(const int label);

public:
    std::vector <conv2dLayer> conv2d_layers;
    std::vector <Layer> layers;
    std::vector <double> firstDenseLayerInput;
    std::vector <double> prediction;
    double get_learning_rate();
    void set_learning_rate(const double lr);

    net(const std::vector <int> & conv2d_filters, const std::vector<int> & layer_sizes, const int input_size);
    void make_prediction(const std::vector<double> & input, const bool dropout = false);

    //double relu(const double x);
    //double d_relu(const double x);
    std::vector<double> softmax(const std::vector<double>& inputs);
    double mean_square_error(const int label);

    std::vector <double> d_mean_square_error(const int label);
    std::vector <double> d_relu_vector (const Layer & layer);
    void calculate_d_mse_for_all_layers(const int label);
    void calculate_d_mse_for_all_fe_layers();
    void adjust_biases();
    void adjust_weights();
    void back_prop(const int label);

    void save_network(std::string conv2d_filename, std::string fc_filename);
    void load_network(std::string fe_filename, std::string fc_filename);

    friend double relu(const double x);
    friend double d_relu(const double x);
};