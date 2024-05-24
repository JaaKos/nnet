#include "net.hpp"

int main()
{
    net myNet({64, 10}, 784);
    myNet.load_network("network/params.csv");

    std::ifstream testfile;
    testfile.open("data/mnist_test.csv");
    std::string line;
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
    std::cout << "Correct: " << correct << std::endl;
    std::cout << "Incorrect: " << incorrect << std::endl;
}