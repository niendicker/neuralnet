
#include <neuralnet.h>

// constructor of neural network class
NeuralNetwork::NeuralNetwork(std::vector<uint> topology, Scalar learningRate)
{
	this->topology = topology;
	this->learningRate = learningRate;
	for (uint i = 0; i < topology.size(); i++) {
		// initialize neuron layers
		if (i == topology.size() - 1)
			neuronLayers.push_back(new RowVector(topology[i]));
		else
			neuronLayers.push_back(new RowVector(topology[i] + 1));

		// initialize cache and delta vectors
		cacheLayers.push_back(new RowVector(neuronLayers.size()));
		deltas.push_back(new RowVector(neuronLayers.size()));

		// vector.back() gives the handle to recently added element
		// coeffRef gives the reference of value at that place
		// (using this as we are using pointers here)
		if (i != topology.size() - 1) {
			neuronLayers.back()->coeffRef(topology[i]) = 1.0;
			cacheLayers.back()->coeffRef(topology[i]) = 1.0;
		}

		// initialize weights matrix
		if (i > 0) {
			if (i != topology.size() - 1) {
				weights.push_back(new Matrix(topology[i - 1] + 1, topology[i] + 1));
				weights.back()->setRandom();
				weights.back()->col(topology[i]).setZero();
				weights.back()->coeffRef(topology[i - 1], topology[i]) = 1.0;
			}
			else {
				weights.push_back(new Matrix(topology[i - 1] + 1, topology[i]));
				weights.back()->setRandom();
			}
		}
	}
};

void NeuralNetwork::propagateForward(RowVector& input)
{
	neuronLayers.front()->block(0, 0, 1, neuronLayers.front()->size() - 1) = input;
	for (uint i = 1; i < topology.size(); i++) {
		(*neuronLayers[i]) = (*neuronLayers[i - 1]) * (*weights[i - 1]);
		neuronLayers[i]->block(0, 0, 1, topology[i]).unaryExpr(std::ptr_fun(activationFunction));
	}
};

void NeuralNetwork::calcErrors(RowVector& targetOutput)
{
  //Output layer
	(*deltas.back()) = targetOutput - (*neuronLayers.back());
  //Hidden layers
	for (uint i = topology.size() - 2; i > 0; i--) {
		(*deltas[i]) = (*deltas[i + 1]) * (weights[i]->transpose());
	}
};

void NeuralNetwork::updateWeights()
{
  // topology.size()-1 = weights.size()
	for (uint layer = 0; layer < topology.size() - 1; layer++) {
		// Hidden layer
    if (layer != topology.size() - 2) {
			for (uint c = 0; c < weights[layer]->cols() - 1; c++) {
				for (uint r = 0; r < weights[layer]->rows(); r++) {
					weights[layer]->coeffRef(r, c) += learningRate * deltas[layer + 1]->coeffRef(c) * activationFunctionDerivative(cacheLayers[layer + 1]->coeffRef(c)) * neuronLayers[layer]->coeffRef(r);
				}
			}
		}
    //output layer
		else {
			for (uint c = 0; c < weights[layer]->cols(); c++) {
				for (uint r = 0; r < weights[layer]->rows(); r++) {
					weights[layer]->coeffRef(r, c) += learningRate * deltas[layer + 1]->coeffRef(c) * activationFunctionDerivative(cacheLayers[layer + 1]->coeffRef(c)) * neuronLayers[layer]->coeffRef(r);
				}
			}
		}
	}
};

void NeuralNetwork::propagateBackward(RowVector& output)
{
	calcErrors(output);
	updateWeights();
};

Scalar NeuralNetwork::activationFunction(Scalar x)
{
  //Using Hyperbolic Tangent (tanh) function
	return tanhf(x);
};

Scalar NeuralNetwork::activationFunctionDerivative(Scalar x)
{
	return 1 - tanhf(x) * tanhf(x);
};


void NeuralNetwork::train(std::vector<RowVector*> input_data, std::vector<RowVector*> output_data)
{
	for (uint i = 0; i < input_data.size(); i++) {
		std::cout << "Input to neural network is : " << *input_data[i] << std::endl;
		propagateForward(*input_data[i]);
		std::cout << "Expected output is : " << *output_data[i] << std::endl;
		std::cout << "Output produced is : " << *neuronLayers.back() << std::endl;
		propagateBackward(*output_data[i]);
		std::cout << "MSE : " << std::sqrt((*deltas.back()).dot((*deltas.back())) / deltas.back()->size()) << std::endl;
	}
};


void ReadCSV(std::string filename, std::vector<RowVector*>& data)
{
	data.clear();
	std::ifstream file;
	std::string line, word;
	// determine number of columns in file
	fgets(line, 256, file);
	std::stringstream ss(line);
	std::vector<Scalar> parsed_vec;
	while (getline(ss, word, ', ')) {
		parsed_vec.push_back(Scalar(std::stof(&word[0])));
	}
	uint cols = parsed_vec.size();
	data.push_back(new RowVector(cols));
	for (uint i = 0; i < cols; i++) {
		data.back()->coeffRef(1, i) = parsed_vec[i];
	}

	// read the file
	if (file.is_open()) {
		while (getline(file, line, '\n')) {
			std::stringstream ss(line);
			data.push_back(new RowVector(1, cols));
			uint i = 0;
			while (getline(ss, word, ', ')) {
				data.back()->coeffRef(i) = Scalar(std::stof(&word[0]));
				i++;
			}
		}
	}
};



