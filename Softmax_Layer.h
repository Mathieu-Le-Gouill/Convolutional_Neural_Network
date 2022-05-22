#pragma once
#include "C_2D_Layer.h"

class Softmax_Layer : public virtual C_2D_Layer
{
	public :
		Softmax_Layer(unsigned inputSize);

		void FeedForward(const std::vector<double>& input, std::vector<double>& output) override;
		void BackPropagate(const std::vector<double>& upstreamLossGradient, std::vector<double>& lossGradient) override;

	private:
		const unsigned m_inputSize;

		std::vector<double> m_output;
};

