#include "Convolution_Layer.h"
#include <iostream>
#include <random>
#include <assert.h>
#include <chrono> 
#include <fstream>
#include <sstream>
#include <string>

using namespace std;

Convolution_Layer::Convolution_Layer(unsigned inputWidth, unsigned inputHeight, unsigned inputDepth, unsigned filterWidth, unsigned filterHeight, unsigned numFilters, unsigned horizontalStride, unsigned verticalStride, const double learningRate)
	: m_inputWidth(inputWidth), m_inputHeight(inputHeight), m_inputDepth(inputDepth), m_filterWidth(filterWidth), m_filterHeight(filterHeight), m_horizontalStride(horizontalStride), m_verticalStride(verticalStride), m_nbFilters(numFilters), m_learningRate(learningRate), m_output(Arma_Cube()), m_input(Arma_Cube()), m_biases(Arma_Cube()), m_outputLossGradients(Arma_Cube())
{
	assert(inputDepth > 0 && numFilters>0 && "Error in Convolution_Layer the input depth and the number of filters must be supperior to 0");
	assert((m_inputWidth - m_filterWidth) % m_horizontalStride == 0 && "Error in the Convolution_Layer constructor, the inputWidth, filterWidth and horizontalStride must be correlated");
	assert((m_inputHeight - m_filterHeight) % m_verticalStride == 0 && "Error in the Convolution_Layer constructor, the inputHeight, filterHeight and verticalStride must be correlated");

	// Compute the output data
	m_outputWidth = (m_inputWidth - m_filterWidth) / m_horizontalStride + 1;
	m_outputHeight = (m_inputHeight - m_filterHeight) / m_verticalStride + 1;
	m_outputDepth = m_nbFilters;

	// Initialize the filters
	m_filters = vector<Arma_Cube>(m_nbFilters, Arma_Cube(filterWidth, filterHeight, inputDepth));

	const int nbInputs = (int)inputWidth * (int)inputHeight * (int)inputDepth;

	for (size_t f = 0; f < m_nbFilters; f++)// For each filters
	{
		auto seed = std::chrono::system_clock::now().time_since_epoch().count();// To get differents epochs 
		std::default_random_engine generator(seed);// Create a generator of random numbers
		std::normal_distribution<double> distribution(0, sqrt(2.0 / nbInputs));// Create a method of distribution of mean 0 and variance sqrt(2/n)

		for (unsigned d = 0; d < m_inputDepth; d++)// For each input depth
		{
			for (unsigned h = 0; h < m_filterHeight; h++)//For each rows of the filter
			{
				for (unsigned w = 0; w < m_filterWidth; w++)//for each element in the row
				{
					double randomValue = distribution(generator);// Generate a random number by the generator using the distribution
					m_filters[f](w,h,d) = randomValue;// Set the filter's weights using ReLu
				}
			}
		}
	}

	//Initialize the biases
	m_biases = Arma_Cube(m_outputWidth, m_outputHeight, m_outputDepth);

}


Convolution_Layer::~Convolution_Layer()
{
}


void Convolution_Layer::FeedForward(Arma_Cube& input, Arma_Cube& output)
{
	assert(input.depth() == m_inputDepth && "Error in the Convolution_Layer FeedForward method, the given input must have the size of the input depth given in the constructor");
	assert(input.height()== m_inputHeight && "Error in the Convolution_Layer FeedForward method, the given input must have the size of the input width given in the constructor");
	assert(input.width() == m_inputWidth && "Error in the Convolution_Layer FeedForward method, the given input must have the size of the input height given in the constructor");

	// Input initialization
	m_input = input;

	// Output initialization
	output = Arma_Cube((m_inputWidth - m_filterWidth) / m_horizontalStride + 1, (m_inputHeight - m_filterHeight) / m_verticalStride + 1, m_nbFilters);


	//-For each output values
	for (unsigned f = 0; f < m_nbFilters; f++)
	{
		for (unsigned h = 0; h < m_outputHeight; h++)
		{
			for (unsigned w = 0; w < m_outputWidth; w++)
			{
				double& result = output(w, h, f);

				//-For each filters values
				for (int d = 0; d < m_inputDepth; d++)
				{
					for (int y = 0; y < m_filterHeight; y++)
					{
						for (int x = 0; x < m_filterWidth; x++)
						{
							// Compute the convolution
							result += m_filters[f](x, y, d) * input(w * m_horizontalStride + x, h * m_verticalStride + y, d);
						}
					}
				}
			}
		}
	}

	// Save the output
	this->m_output = output;
}


void Convolution_Layer::BackPropagate(const Arma_Cube& upstreamLossGradient, Arma_Cube& lossGradient)
{
	assert(upstreamLossGradient.depth() == m_outputDepth && "Error in the Convolution_Layer BackPropagate method, the given upstreamLossGradient must have the size of the output depth given in the constructor");
	assert(upstreamLossGradient.height() == m_outputHeight && "Error in the Convolution_Layer BackPropagate method, the given upstreamLossGradient must have the size of the output width given in the constructor");
	assert(upstreamLossGradient.width() == m_outputWidth && "Error in the Convolution_Layer BackPropagate method, the given upstreamLossGradient must have the size of the output height given in the constructor");

	// Save the upstream loss value
	m_outputLossGradients = upstreamLossGradient;

	// Initialize the loss gradients
	lossGradient = Arma_Cube(m_inputWidth, m_inputHeight, m_inputDepth);

	//- COMPUTE THE LOSS GRADIENT

	// For each upstream gradient values
	for (int f = 0; f < m_nbFilters; f++)
	{
		for (int h = 0; h < m_outputHeight; h++)
		{
			for (int w = 0; w < m_outputWidth; w++)
			{
				const double &upstreamLossValue = upstreamLossGradient(w, h, f);

				// For each filters values
				for (int d = 0; d < m_inputDepth; d++)
				{
					for (int y = 0; y < m_filterHeight; y++)
					{
						for (int x = 0; x < m_filterWidth; x++)
						{
							// Compute the convolution between the flipped filter values and the upstream loss value
							lossGradient(w*m_horizontalStride + x, h*m_verticalStride + y, d) += upstreamLossValue * m_filters[f](x, y, d);
						}
					}
				}
				
			}
		}
	}
}


void Convolution_Layer::Update()
{
	// Initialize the filters gradients 
	m_filtersGradients = vector<Arma_Cube>(m_nbFilters, Arma_Cube(m_filterWidth, m_filterHeight, m_inputDepth));

	//-For each filters values
	for (unsigned f = 0; f < m_nbFilters; f++)
	{
		for (unsigned i = 0; i < m_inputDepth; i++)
		{
			for (unsigned h = 0; h < m_filterHeight; h++)
			{
				for (unsigned w = 0; w < m_filterWidth; w++)
				{
					double& filterGradientValue = m_filtersGradients[f](w, h, i);
					
					//-For each output loss gradient values
					for (int y = 0; y < m_outputHeight; y++)
					{
						for (int x = 0; x < m_outputWidth; x++)
						{
							// Compute the convolution
							filterGradientValue += m_outputLossGradients(x, y, f) * m_input(w * m_horizontalStride + x, h * m_verticalStride + y, i);
						}
					}
				}
			}
		}
	}


	for (size_t f = 0; f < m_nbFilters; f++)// For each filters
	{
		// Update the filters values
		m_filters[f] -= m_filtersGradients[f] * m_learningRate;
	}

	// Update the biases values
	m_biases -= m_outputLossGradients * m_learningRate;
}


void Convolution_Layer::SaveData(const std::string& fileName) const
{
}