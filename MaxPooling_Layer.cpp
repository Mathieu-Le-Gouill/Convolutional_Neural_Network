#include "MaxPooling_Layer.h"
#include <iostream>

using namespace std;

MaxPooling_Layer::MaxPooling_Layer(unsigned inputWidth, unsigned inputHeight, unsigned inputDepth, unsigned poolingSegmentationWidth, unsigned poolingSegmentationHeight, unsigned horizontalStride, unsigned verticalStride)// Constructor
: m_inputWidth(inputWidth), m_inputHeight(inputHeight), m_inputDepth(inputDepth), m_poolingSegmentationWidth(poolingSegmentationWidth), m_poolingSegmentationHeight(poolingSegmentationHeight), m_horizontalStride(horizontalStride), m_verticalStride(verticalStride), m_outputWidth((inputWidth - poolingSegmentationWidth) / horizontalStride + 1), m_outputHeight((inputHeight - poolingSegmentationHeight) / verticalStride + 1), m_outputDepth(inputDepth)
{
	assert((m_inputHeight - m_poolingSegmentationHeight) % m_verticalStride == 0 && "Error in the MaxPooling_Layer FeedForward method, the inputWidth, m_poolingSegmentationHeight and verticalStride must be correlated");
	assert((m_inputWidth - m_poolingSegmentationWidth) % m_horizontalStride == 0 && "Error in the MaxPooling_Layer FeedForward method, the inputWidth, m_poolingSegmentationWidth and horizontalStride must be correlated");
	
	// Initialise the max locations
	m_maxLocations = new bool** [m_inputDepth];
	for (unsigned d = 0; d < m_inputDepth; d++)
	{
		m_maxLocations[d] = new bool* [m_inputHeight];
		for (unsigned h = 0; h < m_inputHeight; h++)
		{
			m_maxLocations[d][h] = new bool[m_inputWidth];
			for (unsigned w = 0; w < m_inputWidth; w++)
			{
				m_maxLocations[d][h][w] = false;
			}
		}
	}
}


MaxPooling_Layer::~MaxPooling_Layer()// Destructor
{
	for (int d = 0; d < m_inputDepth; ++d)
		for (int h = 0; h < m_inputHeight; ++h)
			delete[] m_maxLocations[d][h];

	m_maxLocations = nullptr;
}


void MaxPooling_Layer::FeedForward(Arma_Cube& input, Arma_Cube& output)// Method to reduce the input size using the pooling data
{
	assert(!input.empty() && "Error the given input in MaxPooling_Layer Feedforward method is empty");
	assert(input.width() == m_inputWidth && input.height() == m_inputHeight && input.depth() == m_inputDepth && "Error the given input in MaxPooling_Layer FeedForward method as not the same image size than the input in the constructor");
	
	// Initialize output 
	output = Arma_Cube(m_outputWidth, m_outputHeight, m_outputDepth);

	for (unsigned f = 0; f < m_inputDepth; f++)// For each output depths
	{
		for (unsigned h = 0; h < m_outputHeight; h++)// For each output rows
		{
			for (unsigned w = 0; w < m_outputWidth; w++)// For each output columns
			{
				double maxValue = 0;
				unsigned maxValueLocation[2] = { 0,0 };

				// For each sample values
				for (unsigned y = 0; y < m_poolingSegmentationHeight; y++)
				{
					for (unsigned x = 0; x < m_poolingSegmentationWidth; x++)
					{
						const double& elem = input(w * m_horizontalStride + x, h * m_verticalStride + y, f);

						if (elem > maxValue)
						{
							maxValue = elem;
							maxValueLocation[0] = w * m_horizontalStride + x;
							maxValueLocation[1] = h * m_verticalStride + y;
						}
					}
				}

				// Save the max value location
				m_maxLocations[f][maxValueLocation[1]][maxValueLocation[0]] = true;

				// Save the segmentation sample output
				output(w, h, f) = maxValue;
			}
		}
	}
}


void MaxPooling_Layer::BackPropagate(const Arma_Cube& upstreamLossGradient, Arma_Cube& lossGradient)// Method to compute the layer loss gradient using the upstream loss gradient
{
	assert(upstreamLossGradient.width() == (m_inputWidth - m_horizontalStride) / m_poolingSegmentationWidth + 1 && upstreamLossGradient.height() == (m_inputHeight - m_verticalStride) / m_poolingSegmentationHeight + 1 && upstreamLossGradient.depth() == m_inputDepth &&
	"Error in the MaxPooling_Layer FeedBackward method, the given input have not a correct size");

	// Loss gradient initialisation
	lossGradient = Arma_Cube(m_inputWidth, m_inputHeight, m_inputDepth);

	for (unsigned d = 0; d < m_inputDepth; d++)// For each input depths
	{
		for (unsigned h = 0; h < m_outputHeight; h++)// For each input rows
		{
			for (unsigned w = 0; w < m_outputWidth; w++)// For each input cols
			{
				// For each samples values
				for (unsigned y = 0; y < m_poolingSegmentationHeight; y++)
				{
					for (unsigned x = 0; x < m_poolingSegmentationWidth; x++)
					{
						// Compute the loss gradient using the upstream loss gradient
						if(m_maxLocations[d][h * m_verticalStride + y][w * m_horizontalStride + x])
						lossGradient(w * m_horizontalStride + x, h * m_verticalStride + y, d) = upstreamLossGradient(w, h, d);
					}
				}
			}
		}
	}
}
