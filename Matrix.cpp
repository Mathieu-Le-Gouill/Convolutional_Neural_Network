#include "Matrix.h"
#include <iostream>
#include <assert.h>
#include <math.h>

using namespace std;

Matrix::Matrix()
{
	m_values = nullptr;
	m_nbRows = 0;
	m_nbCols = 0;
}


Matrix::~Matrix()
{
	if (m_values) {
		for (int i = 0; i < m_nbRows; ++i) delete[] m_values[i];
		m_values = nullptr;}
}


Matrix::Matrix(const Matrix& matrix)
{
	const unsigned& nbRows = matrix.m_nbRows;
	const unsigned& nbCols = matrix.m_nbCols;

	this->m_values = new double* [nbRows];

	for (int r = 0; r < nbRows; ++r)
	{
		m_values[r] = new double[nbCols];

		for (int c = 0; c < nbCols; c++)
		{
			this->m_values[r][c] = matrix.m_values[r][c];
		}
	}

	this->m_nbRows = nbRows;
	this->m_nbCols = nbCols;
}


Matrix::Matrix(const vector<vector<double>>& val)
{
	assert(!val.empty() && "Error the given vector in Matrix constructor is empty");
	for (auto i = 1; i < val.size(); ++i) assert(val[i].size() == val[i - 1].size() && "Error in Matrix constructor the given input vector has not an equal size in each of his layers");

	const unsigned& nbRows = val.size();
	const unsigned& nbCols = val.front().size();

	m_nbRows = nbRows;
	m_nbCols = nbCols;

	m_values = new double* [nbRows];
	for (int r = 0; r < nbRows; ++r)
	{
		m_values[r] = new double[nbCols];

		for (int c = 0; c < nbCols; c++)
		{
			m_values[r][c] = val[r][c];
		}
	}
}


Matrix::Matrix(unsigned nbRows, unsigned nbCols, const double* val) : m_nbCols(nbCols), m_nbRows(nbRows)
{
	m_values = new double* [nbRows];
	for (int r = 0; r < nbRows; ++r)
	{
		m_values[r] = new double[nbCols];

		for (int c = 0; c < nbCols; c++)
		{
			m_values[r][c] = *(&val[0] + r*nbCols + c);
		}
	}
}


Matrix::Matrix(unsigned nbRows, unsigned nbCols, double value) : m_nbCols(nbCols), m_nbRows(nbRows)
{
	m_values = new double* [nbRows];
	for (int r = 0; r < nbRows; ++r)
	{
		m_values[r] = new double[nbCols];

		for (int c = 0; c < nbCols; c++)
		{
			m_values[r][c] = value;
		}
	}	
}


void Matrix::print(int approximationCoefficient) const 
{
	const unsigned& nbRows = this->m_nbRows;
	const unsigned& nbCols = this->m_nbCols;

	for (unsigned r = 0; r < nbRows; r++)// For each rows
	{
		for (unsigned c = 0; c < nbCols; c++)// For each columns
		{
			//std::cout << std::scientific;
			std::cout<< ((int)(m_values[r][c] * (double)approximationCoefficient + 0.5)) / (double)approximationCoefficient << " ";// Print the values according to the  to the approximate coefficient, 10 for exemple give the value to the nearest tenth
		}
		std::cout << "\n";
	}
}


bool Matrix::empty() const
{
	return m_nbCols == 0 && m_nbRows == 0;
}


unsigned Matrix::rows() const
{
	return this->m_nbRows;
}


unsigned Matrix::cols() const
{
	return this->m_nbCols;
}


void Matrix::fill(double value)
{
	const unsigned& nbRows = this->m_nbRows;
	const unsigned& nbCols = this->m_nbCols;

	for (int r = 0; r < nbRows; r++)
	{
		for (int c = 0; c < nbCols; c++)
		{
			this->m_values[r][c] = value;
		}
	}
}


Matrix Matrix::transpose() const
{
	const unsigned& nbRows = this->m_nbRows;
	const unsigned& nbCols = this->m_nbCols;

	Matrix result(nbCols, nbRows);

	for (int r = 0; r < nbRows; r++)
	{
		for (int c = 0; c < nbCols; c++)
		{
			result.m_values[c][r] = this->m_values[r][c];
		}
	}
	return std::move(result);
}

Matrix Matrix::flip() const
{
	const unsigned& nbRows = this->m_nbRows;
	const unsigned& nbCols = this->m_nbCols;
	
	Matrix result(nbRows, nbCols);

	for (int r = 0; r < nbRows; r++)
	{
		for (int c = 0; c < nbCols; c++)
		{
			result.m_values[r][c] = this->m_values[nbRows - (r + 1)][nbCols - (c + 1)];
		}
	}
	return std::move(result);
}


double Matrix::scalar(const Matrix& matrix) const
{
	assert(this->m_nbRows == matrix.m_nbRows && this->m_nbCols == matrix.m_nbCols &&
		"Error, the given matrices in scalar method must have the same size");

	const unsigned& nbRows = this->m_nbRows;
	const unsigned& nbCols = this->m_nbCols;

	double result = 0;

	for (int r = 0; r < nbRows; r++)
	{
		for (int c = 0; c < nbCols; c++)
		{
			result += this->m_values[r][c] * matrix.m_values[r][c];
		}
	}

	return result;
}

Matrix Matrix::dot(const Matrix& matrix) const
{
	assert(this->m_nbCols == matrix.m_nbRows &&
		"Error in Matrix dot method, the matrices used must have the relation : first matrix's columns number equal to second matrix's rows number");

	const Matrix& matrixA = *this;
	const Matrix& matrixB = matrix;
	Matrix matrixC;

	matrixC.m_nbRows = matrixA.m_nbRows;
	matrixC.m_nbCols = matrixB.m_nbCols;

	const unsigned& nbRows = matrixC.m_nbRows;
	const unsigned& nbCols = matrixC.m_nbCols;

	double** values = new double* [nbRows];
	for (int r = 0; r < nbRows; ++r)
	{
		values[r] = new double[nbCols];

		for (int c = 0; c < nbCols; c++)
		{
			double sum = 0;
			for (int i = 0; i < matrixA.m_nbCols; i++)
			{
				sum += matrixA.m_values[r][i] * matrixB.m_values[i][c];
			}
			values[r][c] = sum;
		}
	}
	matrixC.m_values = values;

	return std::move(matrixC);
}


Matrix Matrix::subSection(unsigned c, unsigned r, unsigned c2, unsigned r2) const
{
	assert(c <= c2 && r <= r2 && this->m_nbCols > c2 && this->m_nbRows > r2 && 
	"Error in Matrix subSection the given inputs are not contained in the matrix");

	Matrix result((c2-c)+1, (r2-r)+1);

	for (int rows = r; rows <= r2; rows++)
	{
		for (int cols = c; cols <= c2; cols++)
		{
			result.m_values[rows-r][cols-c] = this->m_values[rows][cols];
		}
	}

	return std::move(result);
}


double& Matrix::operator()(unsigned row, unsigned column)
{
	assert(row < m_nbRows && column < m_nbCols && "Error in Matrix operator (), the given index is out of the matrix");
	return this->m_values[row][column];
}


double Matrix::operator()(unsigned row, unsigned column) const
{
	assert(row < m_nbRows&& column < m_nbCols && "Error in Matrix operator (), the given index is out of the matrix");
	return this->m_values[row][column];
}


Matrix Matrix::operator*(double value) const
{
	const unsigned& nbRows = this->m_nbRows;
	const unsigned& nbCols = this->m_nbCols;

	Matrix result = *this;

	for (int r = 0; r < nbRows; r++)
	{
		for (int c = 0; c < nbCols; c++)
		{
			result.m_values[r][c] *= value;
		}
	}
	return std::move(result);
}


Matrix Matrix::operator/(double value) const
{
	const unsigned& nbRows = this->m_nbRows;
	const unsigned& nbCols = this->m_nbCols;

	Matrix result = *this;

	for (int r = 0; r < nbRows; r++)
	{
		for (int c = 0; c < nbCols; c++)
		{
			result.m_values[r][c] /= value;
		}
	}
	return std::move(result);
}


Matrix Matrix::operator^(double value) const
{
	const unsigned& nbRows = this->m_nbRows;
	const unsigned& nbCols = this->m_nbCols;

	Matrix result = *this;

	for (int r = 0; r < nbRows; r++)
	{
		for (int c = 0; c < nbCols; c++)
		{
			result.m_values[r][c] = pow(this->m_values[r][c], value);
		}
	}
	return std::move(result);
}


Matrix Matrix::operator+(const Matrix& matrix) const
{
	assert(this->m_nbRows == matrix.m_nbRows && this->m_nbCols == matrix.m_nbCols &&
		"Error in the Matrix operator+ method, the given matrices have not the same size");

	const unsigned& nbRows = this->m_nbRows;
	const unsigned& nbCols = this->m_nbCols;

	const Matrix& matrixA = *this;
	const Matrix& matrixB = matrix;
	Matrix result = matrixA;

	for (int r = 0; r < nbRows; r++)
	{
		for (int c = 0; c < nbCols; c++)
		{
			result.m_values[r][c] += matrixB.m_values[r][c];
		}
	}

	return std::move(result);
}


Matrix Matrix::operator-(const Matrix& matrix) const
{
	assert(this->m_nbRows == matrix.m_nbRows && this->m_nbCols == matrix.m_nbCols &&
		"Error in Matrix operator- method, the given matrices have not the same size");

	const unsigned& nbRows = this->m_nbRows;
	const unsigned& nbCols = this->m_nbCols;

	const Matrix& matrixA = *this;
	const Matrix& matrixB = matrix;
	Matrix result = matrixA;

	for (int r = 0; r < nbRows; r++)
	{
		for (int c = 0; c < nbCols; c++)
		{
			result.m_values[r][c] -= matrixB.m_values[r][c];
		}
	}

	return std::move(result);
}


Matrix Matrix::operator*(const Matrix& matrix) const
{
	assert(this->m_nbRows == matrix.m_nbRows && this->m_nbCols == matrix.m_nbCols &&
		"Error in Matrix operator*,  the given matrices have not the same size");

	Matrix matrixA = *this;
	const Matrix& matrixB = matrix;

	const unsigned& nbRows = matrixA.m_nbRows;
	const unsigned& nbCols = matrixA.m_nbCols;

	for (int r = 0; r < nbRows; ++r)
	{
		for (int c = 0; c < nbCols; c++)
		{
			matrixA.m_values[r][c] *= matrixB.m_values[r][c];
		}
	}

	return std::move(matrixA);
}


Matrix Matrix::operator/(const Matrix& matrix) const
{
	assert(this->m_nbRows == matrix.m_nbRows && this->m_nbCols == matrix.m_nbCols &&
		"Error in Matrix operator*,  the given matrices have not the same size");

	Matrix matrixA = *this;
	const Matrix& matrixB = matrix;

	const int& nbRows = (int)matrixA.m_nbRows;
	const int& nbCols = (int)matrixA.m_nbCols;

	for (int r = 0; r < nbRows; ++r)
	{
		for (int c = 0; c < nbCols; c++)
		{
			matrixA.m_values[r][c] /= matrixB.m_values[r][c];
		}
	}

	return std::move(matrixA);
}


Matrix& Matrix::operator=(const Matrix& matrix)
{
	const unsigned& nbRows = matrix.m_nbRows;
	const unsigned& nbCols = matrix.m_nbCols;

	this->~Matrix();

	this->m_values = new double*[nbRows];

	for (int r = 0; r < nbRows; ++r)
	{
		m_values[r] = new double[nbCols];

		for (int c = 0; c < nbCols; c++)
		{
			this->m_values[r][c] = matrix.m_values[r][c];
		}
	}

	this->m_nbRows = nbRows;
	this->m_nbCols = nbCols;

	return *this;
}


void Matrix::operator*=(double value)
{
	const unsigned& nbRows = this->m_nbRows;
	const unsigned& nbCols = this->m_nbCols;

	for (int r = 0; r < nbRows; r++)
	{
		for (int c = 0; c < nbCols; c++)
		{
			this->m_values[r][c] *= value;
		}
	}
}


void Matrix::operator/=(double value)
{
	const unsigned& nbRows = this->m_nbRows;
	const unsigned& nbCols = this->m_nbCols;

	for (int r = 0; r < nbRows; r++)
	{
		for (int c = 0; c < nbCols; c++)
		{
			this->m_values[r][c] /= value;
		}
	}
}


void Matrix::operator^=(double value)
{
	const unsigned& nbRows = this->m_nbRows;
	const unsigned& nbCols = this->m_nbCols;

	for (int r = 0; r < nbRows; r++)
	{
		for (int c = 0; c < nbCols; c++)
		{
			this->m_values[r][c] = pow(this->m_values[r][c], value);
		}
	}
}


void Matrix::operator+=(const Matrix& matrix)
{
	assert(this->m_nbRows == matrix.m_nbRows && this->m_nbCols == matrix.m_nbCols &&
		"Error in the Matrix operator+ method, the given matrices have not the same size");

	const unsigned& nbRows = this->m_nbRows;
	const unsigned& nbCols = this->m_nbCols;

	for (int r = 0; r < nbRows; r++)
	{
		for (int c = 0; c < nbCols; c++)
		{
			this->m_values[r][c] += matrix.m_values[r][c];
		}
	}
}


void Matrix::operator-=(const Matrix& matrix)
{
	assert(this->m_nbRows == matrix.m_nbRows && this->m_nbCols == matrix.m_nbCols &&
		"Error in the Matrix operator+ method, the given matrices have not the same size");

	const unsigned& nbRows = this->m_nbRows;
	const unsigned& nbCols = this->m_nbCols;

	for (int r = 0; r < nbRows; r++)
	{
		for (int c = 0; c < nbCols; c++)
		{
			this->m_values[r][c] -= matrix.m_values[r][c];
		}
	}
}


void Matrix::operator*=(const Matrix& matrix)
{
	assert(this->m_nbRows == matrix.m_nbRows && this->m_nbCols == matrix.m_nbCols &&
		"Error in Matrix operator*=,  the given matrices have not the same size");

	const unsigned& nbRows = this->m_nbRows;
	const unsigned& nbCols = this->m_nbCols;

	for (int r = 0; r < nbRows; ++r)
	{
		for (int c = 0; c < nbCols; c++)
		{
			this->m_values[r][c] *=  matrix.m_values[r][c];
		}
	}
}


void Matrix::operator/=(const Matrix& matrix)
{
	assert(this->m_nbRows == matrix.m_nbRows && this->m_nbCols == matrix.m_nbCols &&
		"Error in Matrix operator*=,  the given matrices have not the same size");

	const unsigned& nbRows = this->m_nbRows;
	const unsigned& nbCols = this->m_nbCols;

	for (int r = 0; r < nbRows; ++r)
	{
		for (int c = 0; c < nbCols; c++)
		{
			this->m_values[r][c] /= matrix.m_values[r][c];
		}
	}
}
