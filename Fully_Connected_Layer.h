#pragma once
#include <fstream>
#include <sstream>
#include "C_2D_Layer.h"

class Fully_Connected_Layer : public virtual C_2D_Layer
{
public:

	Fully_Connected_Layer(const std::string& NeuralNetworkfileName, const double learningRate = 1);// Constructor using a network from file
	Fully_Connected_Layer(std::vector<unsigned> networkTopology, const double learningRate = 1);// Constructor to create a new network
	~Fully_Connected_Layer();// Destructor

	void FeedForward(const Matrix &inputsValues, Matrix& outputValues) override;// Method to update the totality of the network according to the inputsValues
	void BackPropagate(const Matrix &upstreamLossGradient, Matrix& lossGradient) override;// Method to propagate the error obtained in the ouput layer to the hiddens
	void Update() override;// Method to update the network according to the error got previously by the back propagation

	void ShowResults(const int accuracy=100) const;// Method to show the current output layer results
	void SaveData(const std::string &fileName) const override;// Method to save the current data progress of the network in a file

private:

	static Matrix Sigmoid(Matrix &matrix);// Sigmoid function for matrix

	void GetData(std::vector<unsigned> &topology, std::vector<double> &weights, std::vector<double> &biases, std::ifstream &networkDataLoading, const std::string &NeuralNetworkfileName);// Method to recover a neural network in a file


	std::vector<unsigned> m_networktopology;// Contains the topology of the network
	std::vector<Matrix> m_outputs;// Contains the neurons activation
	std::vector<Matrix> m_biases;// Contains the neurons biases
	std::vector<Matrix> m_weights;// Contains the neurons weights
	std::vector<Matrix> m_errors;// Contains the errors values

	const double m_learningRate;// Coefficiant corresponding to the learning speed at each epoch

};

