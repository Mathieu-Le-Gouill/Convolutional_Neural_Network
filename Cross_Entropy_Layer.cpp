#include "Cross_Entropy_Layer.h"

Cross_Entropy_Layer::Cross_Entropy_Layer(unsigned inputSize) : m_inputSize(inputSize)
{
}

void Cross_Entropy_Layer::FeedForward(const std::vector<double>& input, std::vector<double>& output)
{
	assert(input.size() == m_inputSize && "Error in FeedForward method in Cross_Entropy_Layer the given input size is not correct");
}


void Cross_Entropy_Layer::BackPropagate(const std::vector<double>& target, std::vector<double>& result)
{
	assert(result.size() == target.size() && "Error in the CrossEntropy function the given inputs are not correct");

	const unsigned& size = result.size();
	std::vector<double> tmp(size);

	for (int i = 0; i < size; i++)
	{
		tmp[i] = result[i] - target[i];
	}

	result = std::move(tmp);
}
