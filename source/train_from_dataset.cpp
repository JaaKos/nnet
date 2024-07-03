#include "net.hpp"
#include <math.h>

int main()
{
    const int epochs = 10;
    const int input_size = 784;

    net myNet({8, 16}, {60, 10}, input_size);

    std::ifstream trainfile;
    std::ifstream testfile;
    std::string line;
    
    for (int i = 0; i < epochs; i++)
    {
        int train_correct = 0;
        int train_incorrect = 0;
        if (!trainfile.is_open()) trainfile.open("data/mnist_train.csv");
        if (!testfile.is_open()) testfile.open("data/mnist_test.csv");

        std::cout << "Epoch " << i+1 << std::endl;
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        while (std::getline(trainfile, line))
        {
            static int count = 0;
            count++;

            if (count % 1000 == 0)
            {
                std::cout << "Correct: " << train_correct << " Incorrect: " << train_incorrect  << std::endl;
                train_correct = 0;
                train_incorrect = 0;
            } 
            std::stringstream csv_line(line);

            std::vector <double> nums;
            nums.resize(784, 0);
            int arrayIndex = 0;

            std::string value;
            std::getline(csv_line, value, ',');
            int label = std::stoi(value);
            while (std::getline(csv_line, value, ','))
            {
                nums[arrayIndex++] = std::stod(value) / 255.0;
            }

            myNet.makePrediction(nums, false);
            myNet.applyBackPropagation(label);

            auto test = myNet.prediction;
            int index = 0;
            double max = 0;
            for (int k = 0; k < test.size(); k++)
            {
                if (test[k] > max) 
                {
                    max = test[k]; 
                    index = k;
                }
            }
            if (index == label) train_correct++;
            else train_incorrect++;
        }

        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;

        int correct = 0;
        int incorrect = 0;

        for (int j = 0; j < 10000; j++)
        {
            std::getline(testfile, line);
            std::stringstream csv_line(line);

            std::vector <double> nums;
            nums.resize(784, 0);
            int arrayIndex = 0;

            std::string value;
            std::getline(csv_line, value, ',');
            int label = std::stoi(value);
            while (std::getline(csv_line, value, ','))
            {
                nums[arrayIndex++] = std::stod(value) / 255.0;
            }

            myNet.makePrediction(nums);

            auto test = myNet.prediction;
            int index = 0;
            double max = 0;
            for (int k = 0; k < test.size(); k++)
            {
                if (test[k] > max) 
                {
                    max = test[k]; 
                    index = k;
                }
            }
            if (index == label) correct++;
            else incorrect++;
        }

        std::cout << std::endl;
        std::cout << "Time taken: " << elapsed_seconds.count() << "s" << std::endl;

        std::cout << "Correct: " << correct << std::endl;
        std::cout << "Incorrect: " << incorrect << std::endl;

        double accuracy = static_cast <double>(correct) / static_cast <double>(correct + incorrect);
        std::cout << std::fixed << std::setprecision(2) << "Accuracy: " << accuracy << std::endl << std::endl;

        trainfile.close();
        testfile.close();

        if (accuracy > 0.9) myNet.setLearningRate(0.9 * myNet.getLearningRate());
    }

    if (trainfile.is_open()) trainfile.close();
    if (testfile.is_open()) testfile.close();

    myNet.saveNetwork("network/fe_params.csv", "network/fc_params.csv");
}