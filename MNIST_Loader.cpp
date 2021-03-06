#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <assert.h>
#include "Arma_Cube.h"

using namespace std;


int ReverseInt(int x);

	vector<Arma_Cube> Load_MNIST_File(const string &MNIST_FilePath, int nbImages)// Function to obtain the data inputs of the images from the MNIST training file
	{
		assert(nbImages <= 60000);

		vector<Arma_Cube> inputsValues(nbImages);

		ifstream file(MNIST_FilePath.c_str(), ios::binary);

		if (file.is_open())
		{
			int magic_number = 0;
			int number_of_images = 0;
			int n_rows = 0;
			int n_cols = 0;

			file.read((char*)&magic_number, sizeof(magic_number));
			magic_number = ReverseInt(magic_number);

			file.read((char*)&number_of_images, sizeof(number_of_images));
			number_of_images = ReverseInt(number_of_images);

			file.read((char*)&n_rows, sizeof(n_rows));
			n_rows = ReverseInt(n_rows);

			file.read((char*)&n_cols, sizeof(n_cols));
			n_cols = ReverseInt(n_cols);

			for (int i = 0; i < nbImages; ++i)
			{
				inputsValues[i] = Arma_Cube(n_cols, n_rows, 1);

				for (int r = 0; r < n_rows; ++r)
				{
					
					for (int c = 0; c < n_cols; ++c)
					{
						unsigned char temp = 0;
						file.read((char*)&temp, sizeof(temp));
						inputsValues[i](c,r,0) = (double)temp / 255.0;
					}
					
				}
			}
		}
		else
			cout << "Error loading MNIST_File..." << endl;

		return move(inputsValues);
	}


	vector<vector<double>> GetTargetValues(const string &LabelFilePath, int nbImages)// Function to obtain the desired output for each images
	{
		vector<vector<double>> targetsValues(nbImages);
		ifstream file(LabelFilePath.c_str(), ios::binary);
		assert(nbImages <= 60000);

		if (file.is_open())
		{
			int magic_number = 0;
			int number_of_images = 0;

			file.read((char*)&magic_number, sizeof(magic_number));
			magic_number = ReverseInt(magic_number);

			file.read((char*)&number_of_images, sizeof(number_of_images));
			number_of_images = ReverseInt(number_of_images);

			for (int i = 0; i < nbImages; ++i)
			{
				targetsValues[i] = vector<double>(10);
				unsigned char temp = 0;
				file.read((char*)&temp, sizeof(temp));

				targetsValues[i][(double)temp] = 1.0;
			}
		}
		else
			cout << "Error loading Label File..." << endl;

		return move(targetsValues);
	}


	int ReverseInt(int x)
	{
		unsigned char ch1, ch2, ch3, ch4;
		ch1 = x & 255;
		ch2 = (x >> 8) & 255;
		ch3 = (x >> 16) & 255;
		ch4 = (x >> 24) & 255;
		return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
	}
