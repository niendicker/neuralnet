#ifndef NEURALNET_H
#define NEURALNET_H

// NeuralNetwork.hpp
#include <Eigen/Dense>
#include <iostream>
#include <vector>

// use typedefs for future ease for changing data types like : float to double
typedef float Scalar;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf ColVector;

using std::vector;

// neural network implementation class!
class NeuralNetwork {
private:
  //Describes how many nodes("neurons") we have in each layer.
  //vector.size() is the number of layers in the network
  vector<uint> topology;
public:
	// storage objects for working of neural network
	/*
		use pointers when using std::vector<Class> as std::vector<Class> calls destructor of
		Class as soon as it is pushed back! when we use pointers it can't do that, besides
		it also makes our neural network class less heavy!! It would be nice if you can use
		smart pointers instead of usual ones like this
		*/
	vector<RowVector*> neuronLayers; // stores the different layers of out network
	vector<RowVector*> cacheLayers; // stores the inactivated (activation fn not yet applied) values of layers
	vector<RowVector*> deltas; // stores the error contribution of each neurons
	vector<Matrix*> weights; // the connection weights itself
	Scalar learningRate;
  
  // constructor
	NeuralNetwork(std::vector<uint> topology, Scalar learningRate = Scalar(0.005));

	// Forward propagation of data
	void propagateForward(RowVector& input);

	// Backward propagation of errors made by neurons
	void propagateBackward(RowVector& output);

	// Calculate errors made by neurons in each layer 
	void calcErrors(RowVector& output);

	// Update the weights of connections
	void updateWeights();

	// Train the neural network give an array of data points
	void train(std::vector<RowVector*> input_data, std::vector<RowVector*> output_data);

//---------------------------------------------------------------------------
//  ANNs use activation functions (AFs) to perform complex computations 
//in the hidden layers and then transfer the result to the output layer. 
//  The primary purpose of AFs is to introduce non-linear properties into ANN.
//They convert the linear input signals of a node into non-linear output signals 
//to improve the learning of high order polynomials that go beyond one degree for
//deep networks. 
//  A UNIQUE aspect of AFs is that they are differentiable
// this helps them function during the BACKPROPAGATION of the neural networks.
//---------------------------------------------------------------------------

  // Forward propagation function
  static Scalar activationFunction(Scalar x);
  // Backward propagation function
  static Scalar activationFunctionDerivative(Scalar x);

};

#endif