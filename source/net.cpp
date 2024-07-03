#include "net.hpp"
#include <iostream>
#include <random>
#include <cmath>
#include <iomanip>

double relu(const double x)
{
    return x > 0 ? x : 0;
}

double d_relu(const double x)
{
    return x > 0 ? 1 : 0;
}


double net::getLearningRate()
{
    return this->learning_rate;
}

void net::setLearningRate(const double lr)
{
    this->learning_rate = lr;
}

net::net(const std::vector<int> & conv2d_filters, const std::vector<int> & layer_sizes, const int input_size)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0.0, 1.0);

    conv2dLayer conv2d_input_layer = {Matrix(28, 28), Matrix(28, 28)};
    Filter filter = {Matrix(3, 3), Matrix(3, 3), 0.01};

    this->conv2d_layers.push_back(conv2d_input_layer);

    for (int i = 0; i < conv2d_filters.size(); i++)
    {
        if (i != 0)
        {
            int nextSize = (conv2d_layers[i - 1].input.getRows() - filter.weights.getRows() + 1) / 2;
            conv2dLayer nextLayer = {Matrix(nextSize, nextSize), Matrix(nextSize, nextSize)};
            this->conv2d_layers.push_back(nextLayer);
        }
        this->conv2d_layers[i].filters.assign(conv2d_filters[i], filter);

        if (i == conv2d_filters.size() - 1)
        {
            this->dense_input_size = (((conv2d_layers[i].input.getRows() - 1) / 2) * ((conv2d_layers[i].input.getColumns() - 1) / 2) * conv2d_filters[i]);
            // std::cout << this->dense_input_size << std::endl;
        }


        for (int j = 0; j < conv2d_filters[i]; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                for (int l = 0; l < 3; l++)
                {
                    this->conv2d_layers[i].filters[j].weights[k][l] = dis(rd);
                }
            }
        }
    }


    for (int i = 0; i < layer_sizes.size(); i++)
    {
        this->layers.push_back({});
        this->layers[i].neurons.assign(layer_sizes[i], {{}, {}, 0, 0, 0});

        for (int j = 0; j < layer_sizes[i]; j++)
        {
            if (i == 0)
            {
                this->layers[i].neurons[j].weights.assign(dense_input_size, 0);
                for (int k = 0; k < dense_input_size; k++)
                {
                    this->layers[i].neurons[j].weights[k] = dis(rd) * std::sqrt(1.0/input_size);
                }
                this->layers[i].neurons[j].bias = 0.1;
            }
            else 
            {
                this->layers[i].neurons[j].weights.assign(layer_sizes[i-1], 0);
                for (int k = 0; k < layer_sizes[i-1]; k++)
                {
                    this->layers[i].neurons[j].weights[k] = dis(rd) * std::sqrt(1.0/layer_sizes[i-1]);
                }
                this->layers[i].neurons[j].bias = 0.1;
            }
        }
    } 
}


void net::makePrediction(const std::vector<double> & input, const bool dropout)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0.0, 1.0);

    this->prediction.clear();
    for (int i = 0; i < this->conv2d_layers.size(); i++)
    {
        this->first_dense_layer_input.clear();
        this->conv2d_layers[i].featuremaps.clear();
        this->conv2d_layers[i].featuremaps_no_activation.clear();
        this->conv2d_layers[i].d_featuremaps.clear();
        this->conv2d_layers[i].d_featuremaps_upsample_positions.clear();
    }

    conv2dLayer & FirstConv2dLayer = this->conv2d_layers[0];
    
    FirstConv2dLayer.input = Matrix(28, 28, input);
    FirstConv2dLayer.input_no_activation = Matrix(28, 28, input);

    FirstConv2dLayer.d_featuremaps_upsample_positions.resize(FirstConv2dLayer.filters.size());
    for (int i = 0; i < FirstConv2dLayer.filters.size(); i++)
    {
        Matrix FeatureMap = FirstConv2dLayer.input.conv2d(FirstConv2dLayer.filters[i].weights);
        FeatureMap = addToEveryCell(FeatureMap, FirstConv2dLayer.filters[i].bias);
        Matrix throwaway;
        Matrix FeatureMapNoActivation = FeatureMap.maxPool(2, 2, throwaway);
        FirstConv2dLayer.featuremaps_no_activation.push_back(FeatureMapNoActivation);
        Matrix withActivation = FeatureMap.forEach(relu).maxPool(2, 2, FirstConv2dLayer.d_featuremaps_upsample_positions[i]);
        FirstConv2dLayer.featuremaps.push_back(withActivation);

        if (this->conv2d_layers.size() == 1)
        {
            std::vector <double> Flattened = FeatureMapNoActivation.flatten();
            this->first_dense_layer_input.insert(this->first_dense_layer_input.end(), Flattened.begin(), Flattened.end());
        }
    }

    for (int i = 1; i < this->conv2d_layers.size(); i++)
    {
        conv2dLayer & CurrentConv2dLayer = this->conv2d_layers[i];
        conv2dLayer & PreviousConv2dLayer = this->conv2d_layers[i - 1];
        CurrentConv2dLayer.input = sumAll(PreviousConv2dLayer.featuremaps);
        CurrentConv2dLayer.input_no_activation = sumAll(PreviousConv2dLayer.featuremaps_no_activation);
        CurrentConv2dLayer.d_featuremaps_upsample_positions.resize(CurrentConv2dLayer.filters.size());
        for (int j = 0; j < CurrentConv2dLayer.filters.size(); j++)
        {
            Matrix FeatureMap = CurrentConv2dLayer.input.conv2d(CurrentConv2dLayer.filters[j].weights);
            FeatureMap = addToEveryCell(FeatureMap, CurrentConv2dLayer.filters[j].bias);
            Matrix throwaway;
            Matrix FeatureMapNoActivation = FeatureMap.maxPool(2, 2, throwaway);
            CurrentConv2dLayer.featuremaps_no_activation.push_back(FeatureMapNoActivation);
            Matrix withActivation = FeatureMap.forEach(relu).maxPool(2, 2, CurrentConv2dLayer.d_featuremaps_upsample_positions[j]);
            CurrentConv2dLayer.featuremaps.push_back(withActivation);

            if (i == this->conv2d_layers.size() - 1)
            {
                std::vector <double> Flattened = FeatureMapNoActivation.flatten();
                this->first_dense_layer_input.insert(this->first_dense_layer_input.end(), Flattened.begin(), Flattened.end());
            }
        }
    }

    std::vector <double> FirstLayerInput(this->first_dense_layer_input.size());
    for (int j = 0; j < this->first_dense_layer_input.size(); j++) FirstLayerInput[j] = relu(this->first_dense_layer_input[j]);
    Layer & FirstLayer = this->layers[0];

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
        FirstLayer.neurons[i].output = relu(FirstLayer.neurons[i].output);
        if (dropout && dis(rd) < this->dropout_rate)
        {
            FirstLayer.neurons[i].output = 0;
        }
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
            if (i != this->layers.size() - 1) this->layers[i].neurons[j].output = relu(this->layers[i].neurons[j].output);
            else this->prediction.push_back(this->layers[i].neurons[j].output);

            if ((dropout && dis(rd) < this->dropout_rate) && i != this->layers.size() - 1)
            {
                this->layers[i].neurons[j].output = 0;
            }
            this->layers[i].neurons[j].input.clear();
        }
    }

    // for (auto i : this->prediction) std::cout << std::fixed << std::setprecision(2) << i << " ";
    // std::cout << std::endl;
    this->prediction = softmax(this->prediction);
    // for (auto i : this->prediction) std::cout << std::fixed << std::setprecision(2) << i << " ";
    // std::cout << std::endl << std::endl;
}

std::vector<double> net::softmax(const std::vector<double>& inputs) 
{
    std::vector<double> exp_values;
    exp_values.reserve(inputs.size());
    double max_input = *std::max_element(inputs.begin(), inputs.end());
    for (double x : inputs) {
        exp_values.push_back(std::exp(x - max_input)); // for numerical stability
    }
    double sum_exp_values = std::accumulate(exp_values.begin(), exp_values.end(), 0.0);
    std::vector<double> outputs;
    outputs.reserve(inputs.size());
    for (double exp_val : exp_values) {
        outputs.push_back(exp_val / sum_exp_values);
    }
    return outputs;
}

double net::calculateMeanSquareError(const int label)
{
    std::vector <double> actual = this->getVectorFromLabel(label);
    double sum = 0;
    for (int i = 0; i < actual.size(); i++)
    {
        sum += (actual[i] - this->prediction[i]) * (actual[i] - this->prediction[i]);
    }
    return sum / actual.size();
}

std::vector <double> net::getVectorFromLabel(const int label)
{
    std::vector <double> vector(10, 0);
    vector[label] = 1;
    return vector;
}

std::vector<double> net::calculateDeltaMSE(const int label)
{
    std::vector <double> actual = this->getVectorFromLabel(label);
    std::vector <double> error(10, 0);
    for (int i = 0; i < actual.size(); i++)
    {
        error[i] = this->prediction[i] - actual[i];
    }
    return error;
}

std::vector <double> net::calculateDeltaReluVector(const Layer & layer)
{
    std::vector <double> z_values_d_relu;
    for (int i = 0; i < layer.neurons.size(); i++)
    {
        z_values_d_relu.push_back(d_relu(layer.neurons[i].z_value));
    }

    return z_values_d_relu;
}

void net::adjustBiases()
{
    for (int i = this->layers.size()-1; i >= 0; i--)
    {
        for (int j = 0; j < this->layers[i].neurons.size(); j++)
        {
            this->layers[i].neurons[j].bias -= this->learning_rate * this->layers[i].mse[j];
        }
    }

    for (int i = 0; i < this->conv2d_layers.size(); i++)
    {
        for (int j = 0; j < this->conv2d_layers[i].d_featuremaps.size(); j++)
        {
            this->conv2d_layers[i].filters[j].bias -= this->learning_rate * this->conv2d_layers[i].d_featuremaps[j].sum();
        }
    }
}

void net::adjustWeights()
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

    for (int i = 0; i < this->conv2d_layers.size(); i++)
    {
        for (int j = 0; j < this->conv2d_layers[i].filters.size(); j++)
        {
            for (int k = 0; k < this->conv2d_layers[i].filters[j].weights.getRows(); k++)
            {
                for (int l = 0; l < this->conv2d_layers[i].filters[j].weights.getColumns(); l++)
                {
                    this->conv2d_layers[i].filters[j].weights[k][l] -= this->learning_rate * this->conv2d_layers[i].filters[j].mse[k][l];
                }
            }
        }
    }
}

void net::calculateDenseLayersDeltaMSE(const int label)
{
    this->layers[layers.size()-1].mse = this->calculateDeltaMSE(label);

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
        std::vector <double> d_relu_values = calculateDeltaReluVector(this->layers[i]);
        std::vector <double> mse_per_neuron;

        for (int j = 0; j < this->layers[i].neurons.size(); j++)
        {
            mse_per_neuron.push_back(error[j] * d_relu_values[j]);
        }

        this->layers[i].mse = mse_per_neuron;
        // for (auto i : this->layers[i].mse) std::cout << std::fixed << std::setprecision(2) << i << " ";
        // std::cout << std::endl;
    }
}

void net::calculateConvFiltersDeltaMSE() //FIXME crashes if layer n has less filters than layer n-1
{
    std::vector <double> error(dense_input_size, 0);
    
    for (int j = 0; j < layers[0].neurons[0].weights.size(); j++)
    {
        for (int k = 0; k < layers[0].neurons.size(); k++)
        {
            error[j] += layers[0].neurons[k].weights[j] * layers[0].mse[k];
            if (layers[0].neurons[k].weights[j] > 100) 
            {
                std::cout << k << " " << j << std::endl;
                std::cout << std::fixed << std::setprecision(2) << layers[0].neurons[k].weights[j] << " ";
                abort();
            }
        }
    }

    conv2dLayer & lastConvLayer = this->conv2d_layers[this->conv2d_layers.size()-1];
    int rowSize = lastConvLayer.input.getRows();
    int colSize = lastConvLayer.input.getColumns();

    int kernelRowSize = lastConvLayer.filters[0].weights.getRows();
    int kernelColSize = lastConvLayer.filters[0].weights.getColumns();

    int conv2dInputSize = dense_input_size / lastConvLayer.filters.size();

    int convRowSize = sqrt(conv2dInputSize);
    int convColSize = convRowSize;

    int convSize = convRowSize * convColSize;

    int filter = 0;
    std::vector <double> mse_per_neuron;

    for (int j = 0; j < first_dense_layer_input.size(); j++)
    {
        mse_per_neuron.push_back(d_relu(first_dense_layer_input[j]) * error[j]);
        // std::cout << dense_input_size << " " << firstDenseLayerInput.size() << " " << error.size() << std::endl;
        if (mse_per_neuron.size() == conv2dInputSize)
        {
            Matrix conv2dmse = Matrix(convRowSize, convColSize, mse_per_neuron).upSample(lastConvLayer.d_featuremaps_upsample_positions[filter]);
            if (lastConvLayer.input_no_activation.getRows() - conv2dmse.getRows() != 2) conv2dmse = conv2dmse.addDimensions(-1, -1);

            lastConvLayer.filters[filter].mse = lastConvLayer.input_no_activation.conv2d(conv2dmse);
            // lastConvLayer.filters[filter].mse.print();
            // std::cout << std::endl;
            lastConvLayer.d_featuremaps.push_back(conv2dmse);
            // lastConvLayer.d_featuremaps_upsample_positions[filter].print();
            // std::cout << std::endl;
            mse_per_neuron.clear();
            filter++;
        }
    }

    for (int i = this->conv2d_layers.size()-2; i >= 0; i--)
    {
        this->conv2d_layers[i].d_featuremaps.resize(this->conv2d_layers[i].filters.size());
        for (int j = 0; j < this->conv2d_layers[i].filters.size(); j++)
        {
            this->conv2d_layers[i].d_featuremaps[j] = conv2d_layers[i+1].d_featuremaps[j].d_conv2d(this->conv2d_layers[i].filters[j].weights);
            Matrix upsampled_fmap = this->conv2d_layers[i].d_featuremaps[j];
            // std::cout << upsampled_fmap.getRows() << " " << conv2d_layers[i].featuremaps_no_activation[j].getRows() << std::endl;
            for (int k = 0; k < upsampled_fmap.getRows(); k++)
            {
                for (int l = 0; l < upsampled_fmap.getColumns(); l++)
                {
                    upsampled_fmap[k][l] *= d_relu(this->conv2d_layers[i].featuremaps_no_activation[j][k][l]);
                }
            }
            upsampled_fmap = upsampled_fmap.upSample(this->conv2d_layers[i].d_featuremaps_upsample_positions[j]);
            this->conv2d_layers[i].filters[j].mse = this->conv2d_layers[i].input_no_activation.conv2d(upsampled_fmap);
            conv2d_layers[i].d_featuremaps[j] = upsampled_fmap;
            // this->conv2d_layers[i].d_featuremaps[j].print();
            // this->conv2d_layers[i].filters[j].mse.print();
            // std::cout << std::endl;
        }
        // system("pause");
    }
}

void net::applyBackPropagation(const int label)
{
    this->calculateDenseLayersDeltaMSE(label);
    this->calculateConvFiltersDeltaMSE();
    this->adjustBiases();
    this->adjustWeights();
    //std::cout << this->conv2d_layers[0].filters.size() << " " << this->conv2d_layers[0].convvector.size() << " " << this->conv2d_layers[0].featuremaps.size() << " " << this->conv2d_layers[0].d_featuremaps.size() << std::endl;
}

void net::saveNetwork(std::string fe_filename, std::string fc_filename)
{
    std::ofstream myfile;
    myfile.open(fe_filename);
    for (int i = 0; i < this->conv2d_layers.size(); i++)
    {
        for (int j = 0; j < this->conv2d_layers[i].filters.size(); j++)
        {
            for (int k = 0; k < this->conv2d_layers[i].filters[j].weights.getRows(); k++)
            {
                for (int l = 0; l < this->conv2d_layers[i].filters[j].weights.getColumns(); l++)
                {
                    if (l == this->conv2d_layers[i].filters[j].weights.getColumns() - 1 && k == this->conv2d_layers[i].filters[j].weights.getRows() - 1)
                    {
                        myfile << this->conv2d_layers[i].filters[j].weights[k][l];
                    }
                    else myfile << this->conv2d_layers[i].filters[j].weights[k][l] << ",";
                }
            }
            myfile << "\n";
        }
    }
    myfile.close();

    myfile.open(fc_filename);
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

void net::loadNetwork(std::string fe_filename, std::string fc_filename)
{
    std::ifstream myfile;
    myfile.open(fe_filename);
    std::string line;

    for (int i = 0; i < this->conv2d_layers.size(); i++)
    {
        for (int j = 0; j < this->conv2d_layers[i].filters.size(); j++)
        {
            std::getline(myfile, line);
            std::stringstream csv_line(line);
            std::string value;
            for (int k = 0; k < this->conv2d_layers[i].filters[j].weights.getRows(); k++)
            {
                for (int l = 0; l < this->conv2d_layers[i].filters[j].weights.getColumns(); l++)
                {
                    std::getline(csv_line, value, ',');
                    this->conv2d_layers[i].filters[j].weights[k][l] = std::stod(value);
                }
            }
        }
    }
    myfile.close();

    myfile.open(fc_filename);

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
