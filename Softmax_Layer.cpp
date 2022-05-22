#include "Softmax_Layer.h"
#include <assert.h>

Softmax_Layer::Softmax_Layer(unsigned inputSize) : m_inputSize(inputSize)
{
}


void Softmax_Layer::FeedForward(const std::vector<double>& input, std::vector<double>& output)
{
	assert(input.size() == m_inputSize && "Error in Softmax_Layer FeedForward method the given input has not the size given in the constructor");

	// Output initialization
	output = std::vector<double> (m_inputSize);
	

	// Compute the sum
	double sum = 0;

	for (const auto& elem : input) {
		sum += exp(elem);
	}

	// Compute the output
	for (int i = 0; i < m_inputSize; ++i) {
		output[i] = exp(input[i])/sum;
	}

	m_output = output;
}


void Softmax_Layer::BackPropagate(const std::vector<double>& upstreamLossGradient, std::vector<double>& lossGradient)
{
	assert(upstreamLossGradient.size() == m_inputSize && "Error in Softmax_Layer BackPropagate method the given upstreamLossGradient has not a correct size");

	// Loss Gradient initialization
	lossGradient = std::vector<double> (m_inputSize);

	/*// Compute the loss gradient
	for (int i = 0; i < m_inputSize; i++)
	{
		double& lossGradientValue = lossGradient[i];
		const double& upstreamLossGradientValue = upstreamLossGradient[i];

		for (int j = 0; j < m_inputSize; j++)
		{
			lossGradientValue += upstreamLossGradientValue  * (1 - upstreamLossGradient[j]);
		}
	}*/

	// For each loss gradient values
	for (int i = 0; i < m_inputSize; i++)
	{
		double& lossGradientValue = lossGradient[i];

		// For each upstream loss gradient values
		for (int j = 0; j < m_inputSize; j++)
		{
			// Compute the loss gradient
			lossGradientValue += (i==j) ? 
			upstreamLossGradient[i] * (1 - upstreamLossGradient[i]) : - upstreamLossGradient[i] * upstreamLossGradient[j];
		}
	}
}
