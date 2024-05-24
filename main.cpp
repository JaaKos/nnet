#include "net.hpp"

int main()
{
    const int epochs = 10;
    const int input_size = 784;

    net myNet({64, 10}, input_size);
    
    std::ifstream trainfile;
    std::ifstream testfile;
    std::string line;

    for (int i = 0; i < epochs; i++)
    {
        if (!trainfile.is_open()) trainfile.open("data/mnist_train.csv");
        if (!testfile.is_open()) testfile.open("data/mnist_test.csv");

        std::cout << "Epoch " << i+1 << std::endl;
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        while (std::getline(trainfile, line))
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
            myNet.back_prop(label);
        }

        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;

        int correct = 0;
        int incorrect = 0;

        for (int j = 0; j < 10000; j++)
        {
            std::getline(testfile, line);
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
        std::cout << "Time taken: " << elapsed_seconds.count() << "s" << std::endl;

        std::cout << "Correct: " << correct << std::endl;
        std::cout << "Incorrect: " << incorrect << std::endl;

        trainfile.close();
        //testfile.close();
    }

    if (testfile.is_open()) testfile.close();
    myNet.save_network("network/params.csv");
}