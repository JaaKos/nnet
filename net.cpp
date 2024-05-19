#include "net.hpp"
#include <iostream>
#include <random>
#include <cmath>


net::net()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (int i = 0; i < 64; i++) this->FirstLayer.neurons.push_back({{}, {}, 0, 0, 0});
    std::vector<double> FirstLayerWeights;
    FirstLayerWeights.assign(784, 0);
    for (int i = 0; i < 64; i++)
    {
        for (int j = 0; j < 784; j++) FirstLayerWeights[j] = dis(rd) * std::sqrt(1.0/784.0);
        this->FirstLayer.neurons[i].weights = FirstLayerWeights;
        this->FirstLayer.neurons[i].bias = dis(rd);
    } 

    for (int i = 0; i < 10; i++) this->OutputLayer.neurons.push_back({{}, {}, 0, 0, 0});
    std::vector<double> OutputLayerweights;
    OutputLayerweights.assign(64, 0);
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 64; j++) OutputLayerweights[j] = dis(rd) * std::sqrt(1.0/64.0);
        this->OutputLayer.neurons[i].weights = OutputLayerweights;
        this->OutputLayer.neurons[i].bias = dis(rd);
    } 

    std::cout << std::endl;
}


void net::make_prediction(const std::array<double, 784> & input)
{
    this->prediction.clear();
    std::vector<double> FirstLayerInput(input.begin(), input.end());

    for (int i = 0; i < this->FirstLayer.neurons.size(); i++) 
    {
        this->FirstLayer.neurons[i].input = FirstLayerInput;
        for (int j = 0; j < this->FirstLayer.neurons[i].input.size(); j++)
        {
            this->FirstLayer.neurons[i].output += this->FirstLayer.neurons[i].input[j] * this->FirstLayer.neurons[i].weights[j];
        }
        this->FirstLayer.neurons[i].output += this->FirstLayer.neurons[i].bias;
        this->FirstLayer.neurons[i].z_value = this->FirstLayer.neurons[i].output;
        this->FirstLayer.neurons[i].output = sigmoid(this->FirstLayer.neurons[i].output);
        this->FirstLayer.neurons[i].input.clear();
    }
    for (int i = 0; i < this->OutputLayer.neurons.size(); i++)
    {
        for (int j = 0; j < this->FirstLayer.neurons.size(); j++)
        {
            this->OutputLayer.neurons[i].input.push_back(this->FirstLayer.neurons[j].output);
        }

        for (int j = 0; j < this->OutputLayer.neurons[i].input.size(); j++)
        {
            this->OutputLayer.neurons[i].output += this->OutputLayer.neurons[i].input[j] * this->OutputLayer.neurons[i].weights[j];
        }
        this->OutputLayer.neurons[i].output += this->OutputLayer.neurons[i].bias;
        this->OutputLayer.neurons[i].z_value = this->OutputLayer.neurons[i].output;
        this->OutputLayer.neurons[i].output = sigmoid(this->OutputLayer.neurons[i].output);
        this->prediction.push_back(this->OutputLayer.neurons[i].output);
        this->OutputLayer.neurons[i].input.clear();
    }
}

double net::sigmoid(const double x)
{
    return 1.0 / (1.0 + std::exp(-x));
}

double net::d_sigmoid(const double x)
{
    return sigmoid(x) * (1.0 - sigmoid(x));
}

double net::mean_square_error(const int label)
{
    std::vector <double> actual = this->label_to_vector(label);
    double sum = 0;
    for (int i = 0; i < actual.size(); i++)
    {
        sum += (actual[i] - this->prediction[i]) * (actual[i] - this->prediction[i]);
    }
    return sum / actual.size();
}

std::vector <double> net::label_to_vector(const int label)
{
    std::vector <double> vector(10, 0);
    vector[label] = 1;
    return vector;
}

std::vector<double> net::d_mean_square_error(const int label)
{
    std::vector <double> actual = this->label_to_vector(label);
    std::vector <double> error(10, 0);
    for (int i = 0; i < actual.size(); i++)
    {
        error[i] = this->prediction[i] - actual[i];
    }
    //for (double i : error) std::cout << i << std::endl;
    return error;
}

std::vector <double> net::d_sigmoid_vector(const Layer & layer)
{
    std::vector <double> z_values_d_sigmoid;
    for (int i = 0; i < layer.neurons.size(); i++)
    {
        z_values_d_sigmoid.push_back(d_sigmoid(layer.neurons[i].z_value));
    }
    //for (double i : z_values_sigmoid) std::cout << i << std::endl;
    return z_values_d_sigmoid;
}

std::vector <double> net::neuron_representation_in_mse(const int label, const Layer & layer, const Layer & next_layer)
{
    std::vector <double> error = d_mean_square_error(label);
    if (next_layer.neurons.size() != 0)
    {
        std::vector <double> new_error(next_layer.neurons[0].weights.size(), 0);
        for (int i = 0; i < next_layer.neurons[0].weights.size(); i++)
        {
            for (int j = 0; j < next_layer.neurons.size(); j++)
            {
                new_error[i] += next_layer.neurons[j].weights[i] * error[j];
            }
        }
        error = new_error;
    }
    //for (double i : error) std::cout << i << std::endl;
    std::vector <double> d_sigmoid_values = d_sigmoid_vector(layer);
    std::vector <double> mse_per_neuron;
    //std::cout << d_sigmoid_values.size() << " " << error.size() << std::endl;

    for (int i = 0; i < layer.neurons.size(); i++)
    {
        mse_per_neuron.push_back(error[i] * d_sigmoid_values[i]);
    }

    //for (double i : mse_per_neuron) std::cout << i << std::endl;
    return mse_per_neuron;
}

void net::adjust_biases(const int label, Layer & layer, const Layer & next_layer)
{
    std::vector <double> layer_mse = neuron_representation_in_mse(label, layer, next_layer);
    for (int i = 0; i < layer.neurons.size(); i++)
    {
        layer.neurons[i].bias -= this->learning_rate * layer_mse[i];
    }
}

void net::adjust_weights(const int label, Layer & layer, const Layer & next_layer)
{
    std::vector <double> layer_mse;
    layer_mse = neuron_representation_in_mse(label, layer, next_layer);
    for (int i = 0; i < layer.neurons.size(); i++)
    {
        for (int j = 0; j < layer.neurons[i].weights.size(); j++)
        {
            layer.neurons[i].weights[j] -= this->learning_rate * layer_mse[i] * layer.neurons[i].input[j];
        }
    }
}
