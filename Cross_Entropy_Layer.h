#pragma once
#include "C_2D_Layer.h"

class Cross_Entropy_Layer : public virtual C_2D_Layer
{
	public:
		Cross_Entropy_Layer(unsigned inputSize);

		void FeedForward(const std::vector<double>& input, std::vector<double>& output) override;
		void BackPropagate(const std::vector<double>& target, std::vector<double>& result);

	private:

		unsigned m_inputSize;
};

