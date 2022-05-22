#include "ReLu_Layer.h"
#include <iostream>
#include <assert.h>

using namespace std;

ReLu_Layer::ReLu_Layer(unsigned inputWidth, unsigned inputHeight, unsigned inputDepth)// Constructor
	:m_inputWidth(inputWidth), m_inputHeight(inputHeight), m_inputDepth(inputDepth)
{
	// Initialize the negative values location
	m_negativeValuesLocations = new bool** [m_inputDepth];
	for (unsigned d = 0; d < m_inputDepth; d++)
	{
		m_negativeValuesLocations[d] = new bool*[m_inputHeight];
		for (unsigned h = 0; h < m_inputHeight; h++)
		{
			m_negativeValuesLocations[d][h] = new bool[m_inputWidth];
			for (unsigned w = 0; w < m_inputWidth; w++)
			{
				m_negativeValuesLocations[d][h][w] = false;
			}
		}
	}
}


ReLu_Layer::~ReLu_Layer()// Destructor
{
	for (int d = 0; d < m_inputDepth; ++d)
		for(int h = 0; h < m_inputHeight; ++h) 
			delete[] m_negativeValuesLocations[d][h];

	m_negativeValuesLocations = nullptr;
}


void ReLu_Layer::FeedForward(Arma_Cube& input, Arma_Cube& output)// Method to keep only the positive values of the input
{
	assert(input.width() == m_inputWidth && input.height() == m_inputHeight && input.depth() == m_inputDepth && 
	"Error the given input in ReLu_Layer FeedForward method have not a correct size");

	// Initialize the output and the positive numbers locations
	output = input;

	for (unsigned d = 0; d < m_inputDepth; d++)// For each input depths
	{
		for (unsigned h = 0; h < m_inputHeight; h++)//For each input rows
		{
			for (unsigned w = 0; w < m_inputWidth; w++)//For each input columns
			{
				auto& value = output(w, h, d);
				if (value < 0.0)// if the output value is negative :
				{
					value = 0.0;// Assign the value to 0
					m_negativeValuesLocations[d][h][w] = true;// Save the negative value location
				}
			}
		}
	}
}

void ReLu_Layer::BackPropagate(const Arma_Cube& upstreamLossGradient, Arma_Cube& lossGradient)// Method to compute the layer loss gradient using the upstream loss gradient
{
	assert(upstreamLossGradient.width() == m_inputWidth && upstreamLossGradient.height() == m_inputHeight && upstreamLossGradient.depth() == m_inputDepth &&
	"Error the given input in ReLu_Layer BackPropagate method have not a correct size");

	// Initialize the output and the positive numbers locations
	lossGradient = upstreamLossGradient ;

	for (unsigned d = 0; d < m_inputDepth; d++)// For each input depths
	{
		for (unsigned h = 0; h < m_inputHeight; h++)//For each input rows
		{
			for (unsigned w = 0; w < m_inputWidth; w++)//For each input columns
			{
				if (m_negativeValuesLocations[d][h][w])// if the previous value location was negative in the forward pass:
				{
					lossGradient(w,h,d) = 0.0;// Assign the value to 0
				}
			}
		}
	}
}
