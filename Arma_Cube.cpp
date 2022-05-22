#include "Arma_Cube.h"
#include <iostream>

using namespace std;

Arma_Cube::Arma_Cube()
	: m_depth(0), m_height(0), m_width(0), m_values(nullptr)
{
}


Arma_Cube::Arma_Cube(unsigned width, unsigned height, unsigned depth, double value)
	: m_depth(depth), m_height(height), m_width(width)
{
	m_values = new Matrix [depth];
	for (int d = 0; d < depth; ++d)
	{
		m_values[d] = Matrix(height, width, value);
	}
}


Arma_Cube::Arma_Cube(const std::vector<Matrix>& values)
{
	assert(!values.empty() && "Error the given vector in Arma_Cube constructor is empty");

	m_depth = values.size();
	m_height = values.front().rows();
	m_width = values.front().cols();

	m_values = new Matrix[m_depth];
	for (int d = 0; d < m_depth; d++)
	{
		m_values[d] = values[d];
	}
}


Arma_Cube::Arma_Cube(const Arma_Cube& cube)
	: m_depth(cube.m_depth), m_height(cube.m_height), m_width(cube.m_width)
{
	const int& depth = (int)cube.m_depth;
	
	m_values = new Matrix[depth];
	for (int d = 0; d < depth; ++d)
	{
		m_values[d] = cube.m_values[d];
	}
}


Arma_Cube::~Arma_Cube()
{
	if (m_values) {
		delete[] m_values;
		m_values = nullptr;}
}


bool Arma_Cube::empty() const
{
	return m_depth == 0 || m_width ==0 || m_height==0;
}


void Arma_Cube::clear()
{
	delete[] m_values;
	m_values = nullptr;

	m_depth = 0, m_height = 0; m_width = 0;
}


Arma_Cube Arma_Cube::flip() const
{
	const int& depth = (int)this->m_depth;

	Arma_Cube result;
	result.m_depth = this->m_depth, result.m_height = this->m_height, result.m_width = this->m_width;

	result.m_values = new Matrix[depth];

	for (int d = 0; d < depth; d++)
	{
		result.m_values[d] = m_values[d].flip();
	}

	return move(result);
}


void Arma_Cube::print(int approximationCoefficient) const
{
	const int& depth = (int)this->m_depth;

	for (int d = 0; d < depth; d++)
	{
		cout << "Layer " << d <<" :"<< endl;

		m_values[d].print(approximationCoefficient);
		cout << "\n";
	}
}


unsigned Arma_Cube::width() const
{
	return m_width;
}


unsigned Arma_Cube::height() const
{
	return m_height;
}


unsigned Arma_Cube::depth() const
{
	return m_depth;
}


void Arma_Cube::fill(double value)
{
	const int& depth = (int)this->m_depth;

	for (int d = 0; d < depth; d++)
	{
		m_values[d].fill(value);
	}
}


Arma_Cube Arma_Cube::subcube(unsigned x, unsigned y, unsigned z, unsigned x2, unsigned y2, unsigned z2) const
{
	assert(x <= x2 && y <= y2 && z <= z2 && this->m_width > x2 && this->m_height > y2 && this->m_depth > z2 &&
		"Error in Arma_Cube subcube method, the given inputs are not contained in the cube");

	Arma_Cube result;
	result.m_depth = (z2 - z) + 1, result.m_height = (y2 - y) + 1, result.m_width = (x2 - x) + 1;

	result.m_values = new Matrix[result.m_depth];

	for (size_t d = z; d <= z2; d++)
	{
		result.m_values[d - z] = this->m_values[d].subSection(x, y, x2, y2);
	}

	return move(result);
}


double Arma_Cube::scalar(const Arma_Cube& cube) const
{
	assert(cube.width() == this->m_width && cube.height() == this->m_height && cube.depth() == this->m_depth &&
		"Error in Arma_Cube dot method the given cube must have the same size");

	double result = 0;
	for (size_t d = 0; d < this->m_depth; d++)
		for (size_t h = 0; h < this->m_height; h++)
			for (size_t w = 0; w < this->m_width; w++)
				result += cube.m_values[d](h,w) * this->m_values[d](h,w);

	return result;
}


Arma_Cube Arma_Cube::operator-(const Arma_Cube& cube) const
{
	assert(this->m_width == cube.m_width && this->m_height == cube.m_height && this->m_depth == cube.m_depth && "Error : the cubes used must have an equal size to use the operator -");
	// Be sure that the both cubes have the same size

	Arma_Cube cubeA = *this;
	const Arma_Cube& cubeB = cube;

	for (size_t d = 0; d < m_depth; d++)
	{
		cubeA.m_values[d] -= cubeB.m_values[d];
	}

	return move(cubeA);
}


Arma_Cube Arma_Cube::operator+(const Arma_Cube& cube) const
{
	assert(this->m_width == cube.m_width && this->m_height == cube.m_height && this->m_depth == cube.m_depth && "Error : the cubes used must have an equal size to use the operator +");
	// Be sure that the both cubes have the same size

	Arma_Cube cubeA = *this;
	const Arma_Cube& cubeB = cube;

	for (size_t d = 0; d < m_depth; d++)
	{
		cubeA.m_values[d] += cubeB.m_values[d];
	}

	return move(cubeA);
}


Arma_Cube Arma_Cube::operator*(const Arma_Cube& cube) const
{
	assert(this->m_width == cube.m_width && this->m_height == cube.m_height && this->m_depth == cube.m_depth && "Error : the cubes used must have an equal size to use the operator *");
	// Be sure that the both cubes have the same size

	Arma_Cube cubeA = *this;
	const Arma_Cube& cubeB = cube;

	for (size_t d = 0; d < m_depth; d++)// For each layer
	{
		cubeA.m_values[d] *= cubeB.m_values[d];
	}

	return move(cubeA);
}


Arma_Cube Arma_Cube::operator/(const Arma_Cube& cube) const
{
	assert(this->m_width == cube.m_width && this->m_height == cube.m_height && this->m_depth == cube.m_depth && "Error : the cubes used must have an equal size to use the operator /");
	// Be sure that the both cubes have the same size

	Arma_Cube cubeA = *this;
	const Arma_Cube& cubeB = cube;

	for (size_t d = 0; d < m_depth; d++)// For each layer
	{
		cubeA.m_values[d] /= cubeB.m_values[d];
	}

	return move(cubeA);
}


Arma_Cube Arma_Cube::operator*(double value) const
{
	Arma_Cube result = *this;

	for (size_t d = 0; d < m_depth; d++)// For each layer
	{
		result.m_values[d] *= value;
	}

	return move(result);
}


Arma_Cube Arma_Cube::operator/(double value) const
{
	Arma_Cube result = *this;

	for (size_t d = 0; d < m_depth; d++)// For each layer
	{
		result.m_values[d] /= value;
	}

	return move(result);
}


Arma_Cube& Arma_Cube::operator=(const Arma_Cube& cube)
{
	if (this == &cube)
		return *this;

	this->~Arma_Cube();
	
	const int& depth = (int)cube.m_depth;

	this->m_values = new Matrix[depth];

	for (int d = 0; d < depth; ++d)
	{
		this->m_values[d] = cube.m_values[d];
	}

	this->m_depth = cube.m_depth, this->m_height = cube.m_height, this->m_width = cube.m_width;

	return *this;
}


void Arma_Cube::operator*=(double value) const
{
	for (size_t d = 0; d < m_depth; d++)// For each layer
	{
		this->m_values[d] *= value;
	}
}


void Arma_Cube::operator/=(double value) const
{
	for (size_t d = 0; d < m_depth; d++)// For each layer
	{
		this->m_values[d] /= value;
	}
}


void Arma_Cube::operator-=(const Arma_Cube& cube)
{
	assert(this->m_width == cube.m_width && this->m_height == cube.m_height && this->m_depth == cube.m_depth && "Error : the cubes used must have an equal size to use the operator -=");
	// Be sure that the both cubes have the same size

	const Arma_Cube& cubeA = *this;
	const Arma_Cube& cubeB = cube;

	for (size_t d = 0; d < m_depth; d++)
	{
		cubeA.m_values[d] -= cubeB.m_values[d];
	}
}


void Arma_Cube::operator+=(const Arma_Cube& cube)
{
	assert(this->m_width == cube.m_width && this->m_height == cube.m_height && this->m_depth == cube.m_depth && "Error : the cubes used must have an equal size to use the operator +=");
	// Be sure that the both cubes have the same size

	const Arma_Cube& cubeA = *this;
	const Arma_Cube& cubeB = cube;

	for (size_t d = 0; d < m_depth; d++)
	{
		cubeA.m_values[d] += cubeB.m_values[d];
	}
}


void Arma_Cube::operator*=(const Arma_Cube& cube)
{
	assert(this->m_width == cube.m_width && this->m_height == cube.m_height && this->m_depth == cube.m_depth && "Error : the cubes used must have an equal size to use the operator *=");
	// Be sure that the both cubes have the same size

	const Arma_Cube& cubeA = *this;
	const Arma_Cube& cubeB = cube;

	for (size_t d = 0; d < m_depth; d++)
	{
		cubeA.m_values[d] *= cubeB.m_values[d];
	}
}


Matrix& Arma_Cube::operator[](unsigned index)
{
	assert(index < this->m_depth && "Error : the given value is out of the cube range...");
	return this->m_values[index];
}


Matrix Arma_Cube::operator[](unsigned index) const
{
	assert(index < this->m_depth && "Error : the given value is out of the cube range...");
	return this->m_values[index];
}


double& Arma_Cube::operator()(unsigned x, unsigned y, unsigned z)
{
	assert(z < this->m_depth && y < this->m_height && x < this->m_width && "Error : the given value is out of the cube range...");
	return m_values[z].m_values[y][x];
}


double Arma_Cube::operator()(unsigned x, unsigned y, unsigned z) const
{
	assert(z < this->m_depth&& y < this->m_height&& x < this->m_width && "Error : the given value is out of the cube range...");

	return this->m_values[z].m_values[y][x];
}
