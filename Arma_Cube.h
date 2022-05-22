#pragma once
#include "Matrix.h"
#include <assert.h>

class Arma_Cube
{
public:
	Arma_Cube();// Constructor
	Arma_Cube(unsigned width, unsigned height, unsigned depth, double value = 0);// Overloaded constructor
	Arma_Cube(const std::vector<Matrix>& values);// Overloaded constructor
	Arma_Cube(const Arma_Cube& cube);// Copy constructor
	~Arma_Cube();// Destructor

	bool empty() const;// Method to check if the object is empty
	void clear();// Method to clear the cube
	Arma_Cube flip() const;// Method to flip the cube at 180°

	void print(int approximationCoefficient = 100) const;//Method to print the cube values

	unsigned width() const;// Method to get the object's width
	unsigned height() const;// Method to get the object's height
	unsigned depth() const;// Method to get the object's depth

	void fill(double value);// Method to fill the cube with a constant

	Arma_Cube subcube(unsigned x, unsigned y, unsigned z, unsigned x2, unsigned y2, unsigned z2) const;// Method to get a section of the cube
	double scalar(const Arma_Cube& cube) const;// Method to get the sum of the product from two cubes


	Arma_Cube operator -(const Arma_Cube& cube) const;// Method to manipulate - operator with a cube
	Arma_Cube operator +(const Arma_Cube& cube) const;// Method to manipulate + operator with a cube
	Arma_Cube operator *(const Arma_Cube& cube) const;// Method to manipulate * operator with a cube
	Arma_Cube operator /(const Arma_Cube& cube) const;// Method to manipulate / operator with a cube

	Arma_Cube operator *(double value) const;// Method to manipulate * operator with a constant
	Arma_Cube operator /(double value) const;// Method to manipulate / operator with a constant

	Arma_Cube& operator =(const Arma_Cube& cube);// Method to manipulate = operator with a cube

	void operator *=(double value) const;// Method to manipulate *= operator with a constant
	void operator /=(double value) const;// Method to manipulate /= operator with a constant

	void operator -=(const Arma_Cube& cube);// Method to manipulate -= operator with a cube
	void operator +=(const Arma_Cube& cube);// Method to manipulate += operator with a cube
	void operator *=(const Arma_Cube& cube);// Method to manipulate *= operator with a cube

	Matrix& operator [](unsigned index);// Method to manipulate [] operator with the object
	Matrix operator [](unsigned index) const;// Method to manipulate [] operator with the object

	double& operator ()(unsigned x, unsigned y, unsigned z);// Method to get the value in the cube
	double operator ()(unsigned x, unsigned y, unsigned z) const;// Method to get the value in the cube
private:

	// Data size
	unsigned m_width;
	unsigned m_height;
	unsigned m_depth;

	Matrix* m_values;// Contains the cube values

};

