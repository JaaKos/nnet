#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <array>
#include <sstream>
#include <chrono>
#include <iomanip>


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

class net
{
    double learning_rate = 0.1;
    std::vector <double> label_to_vector(const int label);

public:
    std::vector <Layer> layers;
    std::vector <double> prediction;

    net(const std::vector<int> & layer_sizes, const int input_size);
    void make_prediction(const std::array<double, 784> & input);

    double sigmoid(const double x);
    double d_sigmoid(const double x);
    double mean_square_error(const int label);

    std::vector <double> d_mean_square_error(const int label);
    std::vector <double> d_sigmoid_vector (const Layer & layer);
    void calculate_d_mse_for_all_layers(const int label);
    void adjust_biases();
    void adjust_weights();
    void back_prop(const int label);

    void save_network(std::string filename);
    void load_network(std::string filename);
};