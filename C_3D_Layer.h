#pragma once
#include <assert.h>
#include "Arma_Cube.h"
#include <string>

class C_3D_Layer
{
	public :
		C_3D_Layer() = default;// Constructor
		virtual ~C_3D_Layer() = default;// Virtual destructor
		virtual void FeedForward(Arma_Cube& input, Arma_Cube& output) {};// Virtual method of layers FeedForward method
		virtual void BackPropagate(const Arma_Cube& upstreamLossGradient, Arma_Cube& lossGradient) {};// Virtual method of layers BackPropagate method

		virtual void Update() {};// Virtual method of layers Update method
		virtual void SaveData(const std::string& fileName) const {};// Virtual method of layers SaveData method
};

