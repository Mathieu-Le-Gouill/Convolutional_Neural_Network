#pragma once
#include "Arma_Cube.h"
#include <tuple>
#include <memory>

#include "Convolution_Layer.h"
#include "ReLu_Layer.h"
#include "MaxPooling_Layer.h"
#include "Fully_Connected_Layer.h"
#include "Softmax_Layer.h"
#include "Cross_Entropy_Layer.h"

class Convolutional_Neural_Network
{
	public:
		Convolutional_Neural_Network(size_t inputWidth, size_t inputHeight, size_t inputDepth, const double learningRate);// Default constructor

		void Add_ConvLayer(size_t filterWidth, size_t filterHeight, size_t numFilters, size_t horizontalStride, size_t verticalStride);// Method to add a new convolution layer
		void Add_ConvLayer(Convolution_Layer convLayer, unsigned outputWidth, unsigned outputHeight, unsigned outputDepth);// Method to add a pregenerated convolution layer
		void Add_ReLu_Layer();// Method to add a new ReLu layer
		void Add_MaxPoolLayer(size_t poolingSegmentationWidth, size_t poolingSegmentationHeight, size_t horizontalStride, size_t verticalStride);// Method to add a new MaxPooling layer
		void Set_OutputLayers(std::vector<unsigned> networkTopology);// Method to add a fully connected layer
		void Set_OutputLayers(Fully_Connected_Layer fullConnectLayer, unsigned outputSize);// Method to add a fully connected layer precreated

		void FeedForward(Arma_Cube& input);// Method to feed the input to the CNN
		void BackPropagate(const std::vector<double>& target);// Method to compute the error using the target values to backpropagate it in the CNN
		void Update();// Method to update the CNN weights and biases

		void ShowResult();// Method to print the current output result
		void SaveData(const std::string& fileName);// Method to save the current data progress of the CNN in a file

	private:

		std::vector<std::unique_ptr<C_3D_Layer>> m_hiddenLayers;// Contains the CNN layers (except the FCL)
		std::vector<std::unique_ptr<C_2D_Layer>> m_outputLayers;// Contains the CNN layers (except the FCL)

		std::vector<std::tuple<unsigned, unsigned, unsigned>> m_layersInputSize;

		std::vector<double> m_result;

		static Matrix Flatten(const Arma_Cube& input);
		static std::vector<double> Vectorize(const Matrix& input);
		static Matrix To_Matrix(const std::vector<double>& input);
		static Arma_Cube To_Cube(const Matrix& input, std::tuple<unsigned, unsigned, unsigned>& inputSize);

		const double m_learningRate;// Coefficiant corresponding to the learning speed at each epoch
};

