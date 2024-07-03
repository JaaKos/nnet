#pragma once
#include <iostream>
#include <vector>
#include <math.h>
#include <iomanip>

class Matrix
{
    private:
        std::vector<std::vector<double>> matrix;
        int rows;
        int columns;
    public:
        Matrix();
        Matrix(int rows, int columns);
        Matrix(std::vector<std::vector<double>> matrix);
        Matrix(int rows, int columns, std::vector<double> elements);

        std::vector<std::vector<double>> getMatrix();
        int getRows() const;
        int getColumns() const;
        int getSize() const;
        void setMatrix(std::vector<std::vector<double>> matrix);
        void print() const;
        void printDimensions() const;
        Matrix expandMatrix(int X, int Y);
        Matrix addDimensions(const int rows, const int columns);
        Matrix invertMatrix();
        Matrix subMatrix(const int X, const int Y, const int rows, const int columns);
        Matrix conv2d(const Matrix & kernel);
        Matrix d_conv2d(Matrix kernel);
        Matrix maxPool(const int X, const int Y, Matrix & output_locations);
        Matrix upSample(Matrix & output_locations);
        Matrix forEach(double (*func)(double));
        double sum();
        std::vector <double> flatten();

        std::vector <double> & operator[](int i);
        friend Matrix sumAll(std::vector<Matrix> matrices);
        friend Matrix addToEveryCell(Matrix & matrix, double val);
};