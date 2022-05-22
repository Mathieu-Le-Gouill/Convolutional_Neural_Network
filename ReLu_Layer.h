#pragma once
#include "C_3D_Layer.h"

class ReLu_Layer : public virtual C_3D_Layer
{
public :

	ReLu_Layer(unsigned inputWidth, unsigned inputHeight, unsigned inputDepth);// Constructor
	~ReLu_Layer();// Destructor

	void FeedForward(Arma_Cube& input, Arma_Cube& output) override; // Method to keep only the positive values of the input
	void BackPropagate(const Arma_Cube& upstreamLossGradient, Arma_Cube& lossGradient) override; // Method to compute the layer loss gradient using the upstream loss gradient

private :

	// Input data
	const unsigned m_inputWidth;
	const unsigned m_inputHeight;
	const unsigned m_inputDepth;

	bool*** m_negativeValuesLocations;// Contains the negative values locations
};

