#pragma once
#include "C_3D_Layer.h"

class Convolution_Layer : public virtual C_3D_Layer
{
public :
	Convolution_Layer(unsigned inputWidth, unsigned inputHeight, unsigned inputDepth, unsigned filterWidth, unsigned filterHeight, unsigned numFilters, unsigned horizontalStride, unsigned verticalStride, const double learningRate = 1);// Default Constructor
	~Convolution_Layer();// Destructor

	void FeedForward(Arma_Cube& input, Arma_Cube& output) override;// FeedForward method
	void BackPropagate(const Arma_Cube& upstreamLossGradient, Arma_Cube& lossGradient) override;// BackPropagate method
	void Update() override;// Update method
	void SaveData(const std::string& fileName) const override;// SaveData method

private :
	// ATTRIBUTES
	// Input data size
	unsigned m_inputWidth;
	unsigned m_inputHeight;
	unsigned m_inputDepth;

	// Output data size
	unsigned m_outputWidth;
	unsigned m_outputHeight;
	unsigned m_outputDepth;

	// Filters data size
	unsigned m_nbFilters;
	unsigned m_filterWidth;
	unsigned m_filterHeight;

	// Strides data
	unsigned m_horizontalStride;
	unsigned m_verticalStride;

	// Components data
	std::vector<Arma_Cube> m_filters;
	Arma_Cube m_biases;

	// Stream data
	Arma_Cube m_input;
	Arma_Cube m_output;

	// Gradients data
	std::vector<Arma_Cube> m_filtersGradients;
	Arma_Cube m_outputLossGradients;

	const double m_learningRate;// Coefficiant corresponding to the learning speed at each epoch
// ATTRIBUTES

// FUNCTIONS

// FUNCTIONS
};

