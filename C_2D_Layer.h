#pragma once
#include <vector>
#include <assert.h>
#include "Matrix.h"
#include <assert.h>
#include <string>

class C_2D_Layer
{
	public :
		C_2D_Layer() = default;// Constructor
		virtual ~C_2D_Layer() = default;// Virtual destructor

		virtual void FeedForward(const std::vector<double>& input, std::vector<double>& output) {};// Virtual method of layers FeedForward method
		virtual void BackPropagate(const std::vector<double>& upstreamLossGradient, std::vector<double>& lossGradient) {};// Virtual method of layers BackPropagate method

		virtual void FeedForward(const Matrix& inputsValues, Matrix& outputValues) {}; // Virtual method of layers FeedForward method
		virtual void BackPropagate(const Matrix& upstreamLossGradient, Matrix& lossGradient) {};// Virtual method of layers BackPropagate method

		virtual void Update() {};// Virtual method of layers Update method
		virtual void SaveData(const std::string& fileName) const {};// Virtual method of layers SaveData method
};

