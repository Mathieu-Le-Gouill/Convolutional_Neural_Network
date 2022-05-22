#pragma once
#include <vector>

class Matrix
{
	friend class Arma_Cube;

public :
	Matrix();
	~Matrix();

	Matrix(const Matrix& matrix);
	Matrix(const std::vector<std::vector<double>>& val);
	Matrix(unsigned nbRows, unsigned nbCols, const double* val);
	Matrix(unsigned nbRows, unsigned nbCols, double value = 0.0);

	void print(int approximationCoefficient = 100)  const;
	bool empty() const;
	unsigned rows() const;
	unsigned cols() const;
	void fill(double value);

	Matrix transpose() const;
	Matrix flip() const;
	double scalar(const Matrix& matrix) const;
	Matrix dot (const Matrix& matrix) const;// Matrix product

	Matrix subSection(unsigned c, unsigned r, unsigned c2, unsigned r2) const;

	double& operator ()(unsigned row, unsigned column);
	double operator ()(unsigned row, unsigned column) const;

	Matrix operator *(double value) const;
	Matrix operator /(double value) const;
	Matrix operator ^(double value) const;

	Matrix operator +(const Matrix& matrix) const;
	Matrix operator -(const Matrix& matrix) const;
	Matrix operator *(const Matrix& matrix) const;// Mutltiplication element by element
	Matrix operator /(const Matrix& matrix) const;// Division element by element

	Matrix& operator =(const Matrix& matrix);

	void operator *=(double value);
	void operator /=(double value);
	void operator ^=(double value);

	void operator +=(const Matrix& matrix);
	void operator -=(const Matrix& matrix);
	void operator *=(const Matrix& matrix);
	void operator /=(const Matrix& matrix);

private:
	double** m_values;
	unsigned m_nbCols;
	unsigned m_nbRows;
};

