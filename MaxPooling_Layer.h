#pragma once
#include "C_3D_Layer.h"

class MaxPooling_Layer : public virtual C_3D_Layer
{
public :
	MaxPooling_Layer(unsigned inputWidth, unsigned inputHeight, unsigned inputDepth, unsigned poolingSegmentationWidth, unsigned poolingSegmentationHeight, unsigned horizontalStride, unsigned verticalStride);// Constructor
	~MaxPooling_Layer();// Destructor

	void FeedForward(Arma_Cube& input, Arma_Cube& output) override;// Method to reduce the input size using the pooling data
	void BackPropagate(const Arma_Cube& upstreamLossGradient, Arma_Cube& lossGradient) override;// Method to compute the layer loss gradient using the upstream loss gradient

private :
	
	// Input data size
	const unsigned m_inputWidth;
	const unsigned m_inputHeight;
	const unsigned m_inputDepth;

	// Output data size
	const unsigned m_outputWidth;
	const unsigned m_outputHeight;
	const unsigned m_outputDepth;

	// Pooling Segmentation data size
	const unsigned m_poolingSegmentationWidth;
	const unsigned m_poolingSegmentationHeight;

	// Stride data
	const unsigned m_horizontalStride;
	const unsigned m_verticalStride;

	bool*** m_maxLocations;// Contains the output max values locations
};

