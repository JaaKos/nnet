#include "net.hpp"
#include <iostream>
#include <random>
#include <cmath>


net::net(const std::vector<int> & layer_sizes, const int input_size)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (int i = 0; i < layer_sizes.size(); i++)
    {
        this->layers.push_back({});
        this->layers[i].neurons.assign(layer_sizes[i], {{}, {}, 0, 0, 0});

        for (int j = 0; j < layer_sizes[i]; j++)
        {
            if (i == 0)
            {
                this->layers[i].neurons[j].weights.assign(input_size, 0);
                for (int k = 0; k < input_size; k++)
                {
                    this->layers[i].neurons[j].weights[k] = dis(rd) * std::sqrt(1.0/input_size);
                }
                this->layers[i].neurons[j].bias = dis(rd);
            }
            else 
            {
                this->layers[i].neurons[j].weights.assign(layer_sizes[i-1], 0);
                for (int k = 0; k < layer_sizes[i-1]; k++)
                {
                    this->layers[i].neurons[j].weights[k] = dis(rd) * std::sqrt(1.0/layer_sizes[i-1]);
                }
                this->layers[i].neurons[j].bias = dis(rd);
            }
        }
    } 
}


void net::make_prediction(const std::array<double, 784> & input)
{
    this->prediction.clear();
    Layer & FirstLayer = this->layers[0];
    std::vector<double> FirstLayerInput(input.begin(), input.end());

    for (int i = 0; i < FirstLayer.neurons.size(); i++) 
    {
        FirstLayer.neurons[i].input = FirstLayerInput;
        FirstLayer.neurons[i].output = 0;
        for (int j = 0; j < FirstLayer.neurons[i].input.size(); j++)
        {
            FirstLayer.neurons[i].output += FirstLayer.neurons[i].input[j] * FirstLayer.neurons[i].weights[j];
        }
        FirstLayer.neurons[i].output += FirstLayer.neurons[i].bias;
        FirstLayer.neurons[i].z_value = FirstLayer.neurons[i].output;
        FirstLayer.neurons[i].output = sigmoid(FirstLayer.neurons[i].output);
        FirstLayer.neurons[i].input.clear();
    }

    for (int i = 1; i < this->layers.size(); i++) 
    {
        for (int j = 0; j < this->layers[i].neurons.size(); j++)
        {
            for (int k = 0; k < this->layers[i-1].neurons.size(); k++)
            {
                this->layers[i].neurons[j].input.push_back(this->layers[i-1].neurons[k].output);
            }

            this->layers[i].neurons[j].output = 0;
            for (int k = 0; k < this->layers[i].neurons[j].input.size(); k++)
            {
                this->layers[i].neurons[j].output += this->layers[i].neurons[j].input[k] * this->layers[i].neurons[j].weights[k];
            }
            this->layers[i].neurons[j].output += this->layers[i].neurons[j].bias;
            this->layers[i].neurons[j].z_value = this->layers[i].neurons[j].output;
            this->layers[i].neurons[j].output = sigmoid(this->layers[i].neurons[j].output);
            if (i == this->layers.size() - 1) this->prediction.push_back(this->layers[i].neurons[j].output);
            this->layers[i].neurons[j].input.clear();
        }
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
    return error;
}

std::vector <double> net::d_sigmoid_vector(const Layer & layer)
{
    std::vector <double> z_values_d_sigmoid;
    for (int i = 0; i < layer.neurons.size(); i++)
    {
        z_values_d_sigmoid.push_back(d_sigmoid(layer.neurons[i].z_value));
    }
    return z_values_d_sigmoid;
}

void net::adjust_biases()
{
    for (int i = this->layers.size()-1; i >= 0; i--)
    {
        for (int j = 0; j < this->layers[i].neurons.size(); j++)
        {
            this->layers[i].neurons[j].bias -= this->learning_rate * this->layers[i].mse[j];
        }
    }
}

void net::adjust_weights()
{
    for (int i = this->layers.size()-1; i >= 0; i--)
    {
        for (int j = 0; j < this->layers[i].neurons.size(); j++)
        {
            for (int k = 0; k < this->layers[i].neurons[j].weights.size(); k++)
            {
                this->layers[i].neurons[j].weights[k] -= this->learning_rate * this->layers[i].mse[j] * this->layers[i].neurons[j].input[k];
            }
        }
    }
}

void net::calculate_d_mse_for_all_layers(const int label)
{
    std::vector<double> error = this->d_mean_square_error(label);
    std::vector <double> d_sigmoid_values = d_sigmoid_vector(this->layers[layers.size()-1]);
    std::vector <double> mse_per_neuron;

    for (int j = 0; j < this->layers[layers.size()-1].neurons.size(); j++)
    {
        mse_per_neuron.push_back(error[j] * d_sigmoid_values[j]);
    }

    this->layers[layers.size()-1].mse = mse_per_neuron;

    for (int i = layers.size()-2; i >= 0; i--)
    {
        std::vector <double> error(layers[i+1].neurons[0].weights.size(), 0);
        for (int j = 0; j < layers[i+1].neurons[0].weights.size(); j++)
        {
            for (int k = 0; k < layers[i+1].neurons.size(); k++)
            {
                error[j] += layers[i+1].neurons[k].weights[j] * layers[i+1].mse[k];
            }
        }
        std::vector <double> d_sigmoid_values = d_sigmoid_vector(this->layers[i]);
        std::vector <double> mse_per_neuron;

        for (int j = 0; j < this->layers[i].neurons.size(); j++)
        {
            mse_per_neuron.push_back(error[j] * d_sigmoid_values[j]);
        }

        this->layers[i].mse = mse_per_neuron;
    }
}

void net::back_prop(const int label)
{
    this->calculate_d_mse_for_all_layers(label);
    this->adjust_biases();
    this->adjust_weights();
}

void net::save_network(std::string filename)
{
    std::ofstream myfile;
    myfile.open(filename);
    for (int i = 0; i < this->layers.size(); i++)
    {
        for (int j = 0; j < this->layers[i].neurons.size(); j++)
        {
            for (int k = 0; k < this->layers[i].neurons[j].weights.size(); k++)
            {
                myfile << this->layers[i].neurons[j].weights[k] << ",";
            }
            myfile << this->layers[i].neurons[j].bias << "\n";
        }
    }
    myfile.close();
}

void net::load_network(std::string filename)
{
    std::ifstream myfile;
    myfile.open(filename);
    std::string line;

    for (int i = 0; i < this->layers.size(); i++)
    {
        for (int j = 0; j < this->layers[i].neurons.size(); j++)
        {
            std::getline(myfile, line);
            std::stringstream csv_line(line);
            std::string value;
            for (int k = 0; k < this->layers[i].neurons[j].weights.size(); k++)
            {
                std::getline(csv_line, value, ',');
                this->layers[i].neurons[j].weights[k] = std::stod(value);
            }
            std::getline(csv_line, value, ',');
            this->layers[i].neurons[j].bias = std::stod(value);
        }
    }
    myfile.close();
}
