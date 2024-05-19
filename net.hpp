#pragma once
#include <vector>
#include <array>

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
};

class net
{
    double learning_rate = 0.1;
    std::vector <double> label_to_vector(const int label);

public:
    Layer FirstLayer;
    Layer OutputLayer;
    std::vector <double> prediction;

    net();
    void make_prediction(const std::array<double, 784> & input);

    double sigmoid(const double x);
    double d_sigmoid(const double x);
    double mean_square_error(const int label);

    std::vector <double> d_mean_square_error(const int label);
    std::vector <double> d_sigmoid_vector (const Layer & layer);
    std::vector <double> neuron_representation_in_mse(const int label, const Layer & layer, const Layer & next_layer);
    void adjust_biases(const int label, Layer & layer, const Layer & next_layer);
    void adjust_weights(const int label, Layer & layer, const Layer & next_layer);
};