#include <iostream>
#include <fstream>
#include <vector>
#include "net.hpp"
#include <iomanip>

#pragma pack(push, 1)
struct BitmapFileHeader {
    uint16_t bfType;
    uint32_t bfSize;
    uint16_t bfReserved1;
    uint16_t bfReserved2;
    uint32_t bfOffBits;
};

struct BitmapInfoHeader {
    uint32_t biSize;
    int32_t biWidth;
    int32_t biHeight;
    uint16_t biPlanes;
    uint16_t biBitCount;
    uint32_t biCompression;
    uint32_t biSizeImage;
    int32_t biXPelsPerMeter;
    int32_t biYPelsPerMeter;
    uint32_t biClrUsed;
    uint32_t biClrImportant;
};
#pragma pack(pop)

int main() {
    const char* filename = "image.bmp";

    net myNet({8, 16}, {60, 10}, 784); // make sure to match the network size
    myNet.loadNetwork("network/fe_params.csv", "network/fc_params.csv");
    int label = 0;
    while (label < 10)
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Error: Unable to open file " << filename << std::endl;
            return 1;
        }

        BitmapFileHeader fileHeader;
        BitmapInfoHeader infoHeader;

        // Read the bitmap file header
        file.read(reinterpret_cast<char*>(&fileHeader), sizeof(fileHeader));
        if (fileHeader.bfType != 0x4D42) {
            std::cerr << "Error: Not a valid BMP file" << std::endl;
            return 1;
        }

        // Read the bitmap info header
        file.read(reinterpret_cast<char*>(&infoHeader), sizeof(infoHeader));

        // Check if it's a 28x28 grayscale bitmap
        if (infoHeader.biWidth != 28 || infoHeader.biHeight != 28 || infoHeader.biBitCount != 8) {
            std::cout << infoHeader.biBitCount << std::endl;
            std::cerr << "Error: Image is not 28x28 grayscale" << std::endl;
            return 1;
        }

        // Move the file pointer to the start of the bitmap data
        file.seekg(fileHeader.bfOffBits, std::ios::beg);

        // Read the pixel data
        std::vector<uint8_t> pixelData(infoHeader.biWidth * infoHeader.biHeight);
        file.read(reinterpret_cast<char*>(pixelData.data()), pixelData.size());

        file.close();

        std::vector <double> nums;

        // Print the pixel values

        const char* shades = " .:-=+*#%@";

        // Print the pixel values
        for (int y = 27; y >= 0; --y) {
            for (int x = 0; x < 28; ++x) {
                int pixel = 255 - pixelData[y * 28 + x];
                nums.push_back(pixel / 255.0);
                char shade = shades[pixel / 26];
                std::cout << shade << " ";
            }
            std::cout << std::endl;
        }

        std::cout << std::endl << std::endl;
        myNet.makePrediction(nums);
        for (auto i : myNet.prediction) std::cout << std::fixed << std::setprecision(2) << i << " ";
        std::cout << std::endl;
        auto maxElementIter = std::max_element(myNet.prediction.begin(), myNet.prediction.end());
        int maxIndex = std::distance(myNet.prediction.begin(), maxElementIter);
        std::cout << "prediction: " << maxIndex << std::endl;
        std::cout << "Enter correct answer: ";
        std::cin >> label;

        if (label < 10) myNet.applyBackPropagation(label);

        std::cout << std::endl << std::endl;
    }

    myNet.saveNetwork("network/fe_params.csv", "network/fc_params.csv");

    return 0;
}
