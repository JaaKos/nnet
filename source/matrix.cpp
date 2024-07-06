#include "matrix.hpp"

Matrix::Matrix() : rows(0), columns(0)
{
    matrix.resize(0, std::vector<double>(0));
}

Matrix::Matrix(int rows, int columns) : rows(rows), columns(columns)
{
    matrix.resize(rows, std::vector<double>(columns));
}

Matrix::Matrix(std::vector<std::vector<double>> matrix) : matrix(matrix)
{
    this->rows = matrix.size();
    this->columns = matrix[0].size();
}

Matrix::Matrix(int rows, int columns, std::vector<double> elements) : rows(rows), columns(columns)
{
    matrix.resize(rows, std::vector<double>(columns));
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            matrix[i][j] = elements[i * columns + j];
        }
    }
}

std::vector<std::vector<double>> Matrix::getMatrix()
{
    return matrix;
}

int Matrix::getRows() const
{
    return rows;
}

int Matrix::getColumns() const
{
    return columns;
}

int Matrix::getSize() const
{
    return rows * columns;
}

void Matrix::printDimensions() const
{
    std::cout << rows << " " << columns << std::endl;
}

Matrix Matrix::addDimensions(const int rows, const int columns)
{
    Matrix newmatrix = this->matrix;
    newmatrix.matrix.resize(this->rows + rows);
    for (int i = 0; i < this->rows + rows; i++)
    {
        newmatrix.matrix[i].resize(this->columns + columns);
    }
    newmatrix.setMatrix(newmatrix.getMatrix());
    return newmatrix;
}

void Matrix::setMatrix(std::vector<std::vector<double>> matrix)
{
    this->rows = matrix.size();
    this->columns = matrix[0].size();
    this->matrix = matrix;
}

void Matrix::print() const
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            std::cout << std::fixed << std::setprecision(2) << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

Matrix Matrix::expandMatrix(int X, int Y)
{
    Matrix newmatrix = Matrix(this->rows + 2*Y, this->columns + 2*X);

    int middle_Y = ceil(newmatrix.getRows() / 2);
    int middle_X = ceil(newmatrix.getColumns() / 2);

    int startpoint_Y = middle_X - floor(this->columns / 2);
    int startpoint_X = middle_Y - floor(this->rows / 2);

    for (int i = startpoint_X; i < this->rows + startpoint_X; i++)
    {
        for (int j = startpoint_Y; j < this->columns + startpoint_Y; j++)
        {
            newmatrix.matrix[i][j] = matrix[i - startpoint_X][j - startpoint_Y];
        }
    }

    return newmatrix;
}

Matrix Matrix::invertMatrix()
{
    Matrix newmatrix = this->matrix;

    for (int i = 0; i < rows; i++)
    {
        std::reverse(newmatrix[i].begin(), newmatrix[i].end());
    }
    std::reverse(newmatrix.matrix.begin(), newmatrix.matrix.end());

    return newmatrix;
}

Matrix Matrix::subMatrix(const int X, const int Y, const int rows, const int columns)
{
    Matrix newmatrix = Matrix(rows, columns);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            newmatrix.matrix[i][j] = matrix[i + X][j + Y];
        }
    }

    return newmatrix;
}

Matrix Matrix::conv2d(const Matrix & kernel)
{
    Matrix newmatrix = Matrix(this->rows - kernel.getRows() + 1, this->columns - kernel.getColumns() + 1);
    for (int i = 0; i < newmatrix.getRows(); i++)
    {
        for (int j = 0; j < newmatrix.getColumns(); j++)
        {
            double sum = 0;
            for (int k = 0; k < kernel.getRows(); k++)
            {
                for (int l = 0; l < kernel.getColumns(); l++)
                {
                    sum += matrix[i + k][j + l] * kernel.matrix[k][l];
                }
            }
            newmatrix.matrix[i][j] = sum;
        }
    }

    return newmatrix;
}

Matrix Matrix::d_conv2d(Matrix kernel)
{
    Matrix padded_matrix = expandMatrix(kernel.getRows() - 1, kernel.getColumns() - 1);
    Matrix inverted_kernel = kernel.invertMatrix();

    Matrix newmatrix = padded_matrix.conv2d(inverted_kernel);

    return newmatrix;
}

std::vector <double> Matrix::flatten()
{
    std::vector <double> flatten;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            flatten.push_back(matrix[i][j]);
        }
    }

    return flatten;
}

Matrix sumAll(std::vector<Matrix> matrices)
{
    Matrix newmatrix = Matrix(matrices[0].getRows(), matrices[0].getColumns());
    for (int i = 0; i < matrices.size(); i++)
    {
        for (int j = 0; j < matrices[i].getRows(); j++)
        {
            for (int k = 0; k < matrices[i].getColumns(); k++)
            {
                newmatrix.matrix[j][k] += matrices[i].matrix[j][k];
            }
        }
    }
    return newmatrix;
}

Matrix Matrix::maxPool(const int X, const int Y, Matrix & outputLocations)
{
    if (X <= 0 || Y <= 0)
    {
        std::cout << "Error: Invalid dimensions" << std::endl;
        return *this;
    }

    outputLocations = Matrix(this->rows, this->columns);

    int padRows = (X - (this->rows % X)) % X;
    int padCols = (Y - (this->columns % Y)) % Y;
    Matrix paddedMatrix = addDimensions(padRows, padCols);
    outputLocations = addDimensions(padRows, padCols);
    Matrix newmatrix = Matrix(paddedMatrix.getRows() / X, paddedMatrix.getColumns() / Y);
    for (int i = 0; i < newmatrix.getRows(); i++)
    {
        for (int j = 0; j < newmatrix.getColumns(); j++)
        {
            double max = paddedMatrix.matrix[i * X][j * Y];
            double maxIndex_X = i*X;
            double maxIndex_Y = j*Y;
            for (int k = 0; k < X; k++)
            {
                for (int l = 0; l < Y; l++)
                {
                    if (paddedMatrix.matrix[i * X + k][j * Y + l] > max)
                    {
                        max = paddedMatrix.matrix[i * X + k][j * Y + l];
                        maxIndex_X = i*X + k;
                        maxIndex_Y = j*Y + l;
                    }
                    outputLocations.matrix[i * X + k][j * Y + l] = 0;
                }
            }
            newmatrix.matrix[i][j] = max;
            outputLocations.matrix[maxIndex_X][maxIndex_Y] = 1;
        }
    }

    return newmatrix;
}

Matrix Matrix::upSample(Matrix & output_locations)
{
    Matrix newmatrix = Matrix(output_locations.getRows(), output_locations.getColumns());
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            int posX = i*2;
            int posY = j*2;
            newmatrix.matrix[posX][posY] = matrix[i][j] * output_locations.matrix[posX][posY];
            newmatrix.matrix[posX + 1][posY] = matrix[i][j] * output_locations.matrix[posX + 1][posY];
            newmatrix.matrix[posX][posY + 1] = matrix[i][j] * output_locations.matrix[posX][posY + 1];
            newmatrix.matrix[posX + 1][posY + 1] = matrix[i][j] * output_locations.matrix[posX + 1][posY + 1];
        }
    }

    return newmatrix;
}

double Matrix::sum()
{
    double sum = 0;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            sum += matrix[i][j];
        }
    }
    return sum;
}
Matrix Matrix::forEach(double (*func)(double))
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            matrix[i][j] = func(matrix[i][j]);
        }
    }
    return *this;
}

Matrix addToEveryCell(Matrix & matrix, double val) 
{
    for (int i = 0; i < matrix.getRows(); i++)
    {
        for (int j = 0; j < matrix.getColumns(); j++)
        {
            matrix.matrix[i][j] += val;
        }
    }
    return matrix;
}

std::vector<double> & Matrix::operator[](int i)
{
    return matrix[i];
}

