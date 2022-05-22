#include "Convolutional_Neural_Network.h"
#include <fstream>
#include <iostream>
#include <string>

Convolutional_Neural_Network::Convolutional_Neural_Network(unsigned inputWidth, unsigned inputHeight, unsigned inputDepth, const double learningRate)// Default constructor
	:m_learningRate(learningRate), m_result(std::vector<double>())
{
	m_layersInputSize.push_back({ inputWidth, inputHeight, inputDepth });
}


void Convolutional_Neural_Network::Add_ConvLayer(unsigned filterWidth, unsigned filterHeight, unsigned numFilters, unsigned horizontalStride, unsigned verticalStride)// Method to add a new convolution layer
{
	// Initialization input data size
	const auto [inputWidth, inputHeight, inputDepth] = m_layersInputSize.back();

	assert((inputWidth - filterWidth) % horizontalStride == 0 && "Error in the Convolutional_Neural_Network Add_ConvLayer method, the inputWidth, filterWidth and horizontalStride must be correlated");
	assert((inputHeight - filterHeight) % verticalStride == 0 && "Error in the Convolutional_Neural_Network Add_ConvLayer method, the inputHeight, filterHeight and verticalStride must be correlated");

	// Add the new convolution layer
	m_hiddenLayers.push_back(std::unique_ptr<Convolution_Layer>(new Convolution_Layer(inputWidth, inputHeight, inputDepth, filterWidth, filterHeight, numFilters, horizontalStride, verticalStride, m_learningRate)));

	// Prepare the output size for the next layer
	m_layersInputSize.push_back({ (inputWidth - filterWidth) / horizontalStride + 1, (inputHeight - filterHeight) / verticalStride + 1, numFilters });
}


void Convolutional_Neural_Network::Add_ConvLayer(Convolution_Layer convLayer, unsigned outputWidth, unsigned outputHeight, unsigned outputDepth)// Method to add a pregenerated convolution layer
{
	// Add the new convolution layer
	m_hiddenLayers.push_back(std::make_unique<Convolution_Layer>(std::move(convLayer)));

	// Prepare the output size for the next layer
	m_layersInputSize.push_back({ outputWidth, outputHeight, outputDepth });
}


void Convolutional_Neural_Network::Add_ReLu_Layer()// Method to add a new ReLu layer
{
	// Initialization input data size
	const auto [inputWidth, inputHeight, inputDepth] = m_layersInputSize.back();

	// Add the new ReLu layer
	m_hiddenLayers.push_back(std::unique_ptr<ReLu_Layer>(new ReLu_Layer(inputWidth, inputHeight, inputDepth)));

	// Prepare the output size for the next layer
	m_layersInputSize.push_back({ inputWidth, inputHeight, inputDepth });
}

void Convolutional_Neural_Network::Add_MaxPoolLayer(size_t poolingSegmentationWidth, size_t poolingSegmentationHeight, size_t horizontalStride, size_t verticalStride)// Method to add a new MaxPooling layer
{
	// Initialization input data size
	const auto [inputWidth, inputHeight, inputDepth] = m_layersInputSize.back();

	assert((inputHeight - poolingSegmentationHeight) % verticalStride == 0 && "Error in the Convolutional_Neural_Network Add_MaxPoolLayer method, the inputWidth, m_poolingSegmentationHeight and verticalStride must be correlated");
	assert((inputWidth - poolingSegmentationWidth) % horizontalStride == 0 && "Error in the Convolutional_Neural_Network Add_MaxPoolLayer method, the inputWidth, m_poolingSegmentationWidth and horizontalStride must be correlated");

	// Add the new MaxPooling layer
	m_hiddenLayers.push_back(std::unique_ptr<MaxPooling_Layer>(new MaxPooling_Layer(inputWidth, inputHeight, inputDepth, poolingSegmentationWidth, poolingSegmentationHeight, horizontalStride, verticalStride)));

	// Prepare the output size for the next layer
	m_layersInputSize.push_back({ (inputWidth - poolingSegmentationWidth) / horizontalStride + 1, (inputHeight - poolingSegmentationHeight) / verticalStride + 1, inputDepth });
}

void Convolutional_Neural_Network::Set_OutputLayers(std::vector<unsigned> networkTopology)// Method to add a fully connected layer
{
	// Initialization input data size
	const auto [inputWidth, inputHeight, inputDepth] = m_layersInputSize.back();

	assert(networkTopology.size() >= 2 && "Error : the given network topology in Convolutional_Neural_Network Set_OutputLayers method, require a size greater than 2, for an input layer and an output layer.");
	assert(networkTopology[0] == inputWidth * inputHeight * inputDepth && "Error : the given network topology in Convolutional_Neural_Network Set_OutputLayers method, have not a correct size to accept the previous layer output");

	// Add the new fully connected layer
	m_outputLayers.push_back(std::unique_ptr<Fully_Connected_Layer>(new Fully_Connected_Layer(networkTopology, m_learningRate)));

	// Add the others output layers
	m_outputLayers.push_back(std::unique_ptr<Softmax_Layer>(new Softmax_Layer(networkTopology.back())));
	m_outputLayers.push_back(std::unique_ptr<Cross_Entropy_Layer>(new Cross_Entropy_Layer(networkTopology.back())));
}


void Convolutional_Neural_Network::Set_OutputLayers(Fully_Connected_Layer fullConnectLayer, unsigned outputSize)
{
	m_outputLayers.push_back(std::make_unique<Fully_Connected_Layer>(std::move(fullConnectLayer)));

	// Add the others output layers
	m_outputLayers.push_back(std::unique_ptr<Softmax_Layer>(new Softmax_Layer(outputSize)));
	m_outputLayers.push_back(std::unique_ptr<Cross_Entropy_Layer>(new Cross_Entropy_Layer(outputSize)));
}


void Convolutional_Neural_Network::FeedForward(Arma_Cube& input)// Method to feed the input to the CNN
{
	// Initialization input data size
	const auto [inputWidth, inputHeight, inputDepth] = m_layersInputSize.front();

	assert(!m_hiddenLayers.empty() && !m_outputLayers.empty() && "Error in Convolutional_Neural_Network FeedForward method, the CNN layers have not been created and are empty");
	assert(input.width() == inputWidth && input.height() == inputHeight && input.depth() == inputDepth && "Error in Convolutional_Neural_Network FeedForward method, the given input have not the same size as given in the class constructor");

	// Initialize the hidden layers data
	std::vector<Arma_Cube> outputHiddenData(m_hiddenLayers.size() + 1);
	outputHiddenData.front() = input;

	for (size_t l = 0; l < m_hiddenLayers.size(); l++)// For each layers
	{
		// Feed the input in the layer to compute the new output
		m_hiddenLayers[l]->FeedForward(outputHiddenData[l], outputHiddenData[l + 1]);
	}
	// Feed the fully connected layer and the sofmax layer
	Matrix out;
	m_outputLayers.front()->FeedForward(Flatten(outputHiddenData.back()), out);
	m_outputLayers[1]->FeedForward(Vectorize(out), m_result);
}


void Convolutional_Neural_Network::BackPropagate(const std::vector<double>& target)// Method to compute the error using the target values to backpropagate it in the CNN
{
	// Backpropagate the cross entropy layer
	auto result = m_result;
	m_outputLayers.back()->BackPropagate(target, result);

	// Backpropagate the soft max layer
	std::vector<double> gradient;
	m_outputLayers[1]->BackPropagate(result, gradient);

	// BackPropagate the fully connected layer
	Matrix out;
	m_outputLayers.front()->BackPropagate(To_Matrix(gradient), out);

	// Initialize the hidden layers data
	std::vector<Arma_Cube> outputHiddenData(m_hiddenLayers.size()+1);
	outputHiddenData.back() = To_Cube(out, m_layersInputSize[m_hiddenLayers.size()]);

	for (int l = m_hiddenLayers.size() - 1; l >= 0; l--)// For each layers
	{
		// BackPropagate the input in the layer to compute the new output
		m_hiddenLayers[l]->BackPropagate(outputHiddenData[l + 1], outputHiddenData[l]);
	}
}

void Convolutional_Neural_Network::Update()// Method to update the CNN weights and biases
{
	int i = 0;
	for (auto& hiddenLayer : m_hiddenLayers)// For each hidden layers
	{
		hiddenLayer->Update();// Update the layer
		i++;
	}
		
	for (auto& outputLayer : m_outputLayers)// For each output layers
		outputLayer->Update();// Update the layer
}


void Convolutional_Neural_Network::ShowResult()// Method to print the current output result
{
	std::cout << "Result : ";
	for (int i = 0; i < m_result.size(); i++)
		std::cout << ((double)((int)(m_result[i] * 100.0 + 0.5)) / 100.0) * 100 << " ";
	std::cout << "\n";
}


void Convolutional_Neural_Network::SaveData(const std::string& fileName)// Method to save the current data progress of the CNN in a file
{
	std::ofstream cnnDataSaving(fileName.c_str());// File writer
	if (cnnDataSaving)// If it is open
	{
		// Save the CNN data in the file
		for (size_t l = 0; l < m_hiddenLayers.size(); l++)// For each layers
			m_hiddenLayers[l]->SaveData(fileName + std::to_string(l));// Save the layer data

		m_outputLayers.front()->SaveData(fileName);// Save the fully connected layer data
	}
	else // If it failed to open
		std::cout << "Error, can't oppening file network_data.txt" << std::endl;// Error
}



Matrix Convolutional_Neural_Network::Flatten(const Arma_Cube& input)
{
	assert(!input.empty() && "Error in Flatten function, the given input is empty");

	const unsigned& depth = input.depth();
	const unsigned& height = input.height();
	const unsigned& width = input.width();

	Matrix output(1, width * height * depth);
	for (unsigned d = 0; d < depth; ++d)
	{
		for (unsigned h = 0; h < height; ++h)
		{
			for (unsigned w = 0; w < width; ++w)
			{
				output(0, w + h * width + d * height * width) = input(w, h, d);
			}
		}
	}

	return std::move(output);
}


std::vector<double> Convolutional_Neural_Network::Vectorize(const Matrix& input)
{
	assert(!input.empty() && "Error in Flatten function, the given input is empty");

	const unsigned& width = input.cols();
	const unsigned& height = input.rows();

	std::vector<double> output(width * height);

	for (unsigned h = 0; h < height; h++)
	{
		for (unsigned w = 0; w < width; w++)
		{
			output[w + h * width] = input(h, w);
		}
	}

	return move(output);
}


Matrix Convolutional_Neural_Network::To_Matrix(const std::vector<double>& input)
{
	std::vector<std::vector<double>> v(1);
	v[0] = input;
	return std::move(Matrix(v));
}


Arma_Cube Convolutional_Neural_Network::To_Cube(const Matrix& input, std::tuple<unsigned, unsigned, unsigned>& inputSize)
{
	auto [width, height, depth] = inputSize;
	assert(input.rows() == 1 && input.cols() == width * height * depth && "Error in the To_Cube function the given input are not correct");
	Arma_Cube output(width, height, depth);

	for (unsigned d = 0; d < depth; d++)
	{
		for (unsigned h = 0; h < height; h++)
		{
			for (unsigned w = 0; w < width; w++)
			{
				output(w, h, d) = input(0, w + h * width + d * width * height);
			}
		}
	}

	return std::move(output);
}