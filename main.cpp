#include <iostream>
#include "Convolutional_Neural_Network.h"

using namespace std;

// VARIABLES

const double learningRate = 0.1;
const string dataFileName = "DataFile.txt";

const unsigned nbTrainingImages = 10000;// 60000 max
const unsigned imageWidth = 28, imageHeight = 28;
const unsigned imageDepth = 1;
const unsigned nbDataImage = imageWidth * imageHeight;

// Convolution Layer
const unsigned numFilters = 6;
const unsigned filterWidth = 5, filterHeight = 5;
const unsigned horizontalStride = 1, verticalStride = 1;

// ReLu Layer

// MaxPooling Layer
const unsigned segmentationWidth = 2, segmentationHeight = 2;
const unsigned poolHorizontalStride = 2, poolVerticalStride = 2;

// Fully Connected Layer
const vector<unsigned> fullyConnectedLayerTopology({ 4 * 4 * 16, 84, 10 });

//VARIABLES

//FUNCTIONS DECLARATIONS
vector<Arma_Cube> Load_MNIST_File(const string& MNIST_FilePath, int nbImages);// Function to obtain the data inputs of the images from the MNIST training file
vector<vector<double>> GetTargetValues(const string& LabelFilePath, int nbImages);// Function to obtain the desired output for each images
Matrix Flatten(const Arma_Cube& input);
vector<double> Vectorize(const Matrix& input);
Matrix To_Matrix(const vector<double>& input);
Arma_Cube To_Cube(const Matrix& input, unsigned width, unsigned height, unsigned depth);
void Print(const vector<double>& result, const string& message);
vector<double> CrossEntropyDerivative(const vector<double>& result, const vector<double>& target);
//FUNCTIONS DECLARATIONS

int main()
{
	vector<Arma_Cube> inputsValues = Load_MNIST_File("train-images.idx3-ubyte", nbTrainingImages);// t10k-images.idx3-ubyte OR train-images.idx3-ubyte
	vector<vector<double>> targetsValues = GetTargetValues("train-labels.idx1-ubyte", nbTrainingImages);// t10k-labels.idx1-ubyte OR train-labels.idx1-ubyte

	Convolutional_Neural_Network cnn(imageWidth, imageHeight, imageDepth, learningRate);
	cnn.Add_ConvLayer(filterWidth, filterHeight, numFilters, horizontalStride, verticalStride);
	cnn.Add_ReLu_Layer();
	cnn.Add_MaxPoolLayer(segmentationWidth, segmentationHeight, poolHorizontalStride, poolVerticalStride);

	cnn.Add_ConvLayer(filterWidth, filterHeight, 16, horizontalStride, verticalStride);
	cnn.Add_ReLu_Layer();
	cnn.Add_MaxPoolLayer(segmentationWidth, segmentationHeight, poolHorizontalStride, poolVerticalStride);

	cnn.Set_OutputLayers(fullyConnectedLayerTopology);

	for (auto i = 0; i < nbTrainingImages; i++)// For each epochs
	{
		cout << "Pass " << i << " :" << endl;
		cnn.FeedForward(inputsValues[i]);
		cout << "Target : "; for (auto& val : targetsValues[i]) cout << val << " "; cout << "\n";
		cnn.ShowResult();
		cnn.BackPropagate(targetsValues[i]);
		cnn.Update();
	}

	system("pause");
}

