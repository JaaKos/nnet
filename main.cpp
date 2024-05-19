#include "net.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <array>
#include <sstream>
#include <chrono>

int main()
{
    net myNet;
    std::ifstream file;
    file.open("data/mnist_train.csv");

    std::string line;
    std::chrono::time_point<std::chrono::system_clock> start, end;

    start = std::chrono::system_clock::now();
    while (std::getline(file, line))
    {
        std::stringstream csv_line(line);

        std::array <double, 784> nums;
        int arrayIndex = 0;

        std::string value;
        std::getline(csv_line, value, ',');
        int label = std::stoi(value);
        while (std::getline(csv_line, value, ','))
        {
            nums[arrayIndex++] = std::stod(value) / 255.0;
        }

        myNet.make_prediction(nums);
        myNet.adjust_biases(label, myNet.OutputLayer, {});
        myNet.adjust_weights(label, myNet.OutputLayer, {});
        myNet.adjust_biases(label, myNet.FirstLayer, myNet.OutputLayer);
        myNet.adjust_weights(label, myNet.FirstLayer, myNet.OutputLayer);
    }

    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;

    file.close();
    file.open("data/mnist_test.csv");

    for (int i = 0; i < 10; i++)
    {
        std::getline(file, line);
        std::stringstream csv_line(line);

        std::array <double, 784> nums;
        int arrayIndex = 0;

        std::string value;
        std::getline(csv_line, value, ',');
        int label = std::stoi(value);
        while (std::getline(csv_line, value, ','))
        {
            nums[arrayIndex++] = std::stod(value) / 255.0;
        }

        myNet.make_prediction(nums);

        auto test = myNet.prediction;
        std::cout << label << std::endl;
        for (double i : test) std::cout << i << std::endl;
    }

    std::cout << std::endl << "Time taken: " << elapsed_seconds.count() << "s" << std::endl;
    
}