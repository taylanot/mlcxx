/**
 * @file ann2.cpp
 * @author Ozgur Taylan Turan
 *  This is a simple program that showcases the wrapper for the ANN class for 
 *  regression task
 */

#include <lcpp.h>

// Since it is a regression task and we will train an ANN we use the template 
// parameter arma::Mat<size_t>. DTYPE is "double" by default for some mlpack 
// learners do not have the option to be run in "float" unfortunately.
using DATASET = data::Dataset<arma::Mat<DTYPE>,DTYPE>;
// Let's choose an optimizer Adam from ensmallen.
using OPT = ens::Adam;
// Loss measure is Mean Squared Error (MSE) since we will solve a regression
// problem. These measures should be from mlpack ann module
using LOSS = mlpack::MSE;
// The wrapper for artificial neural-network from lcpp
// Let's choose a network type from mlpack.
using NetworkType = mlpack::FFN<mlpack::MeanSquaredError>;
// Now, declaring the model...
using MODEL = algo::ANN<NetworkType,OPT,LOSS>;


int main ( ) 
{
  // We will be generating our own data with banana dataset aka moons by 
  // sklearn people
  DATASET trainset(2); // creating a 2 dimensional training set
  DATASET testset(2); // creating a 2 dimensional test set
  trainset.Linear(1000,0.1); // We are drawing 1000 samples for training
  testset.Linear(1000,0.1); // We are drawing 1000 samples for testing
                       
  // Let's create the network architecture.
  NetworkType network;
  network.Add<mlpack::Linear>(5);
  network.Add<mlpack::ReLU>();
  network.Add<mlpack::Linear>(5);
  network.Add<mlpack::ReLU>();
  network.Add<mlpack::Linear>(1);

  // Initialize the model with inputs, and labels 
  MODEL ann(trainset.inputs_,trainset.labels_,network);

  LOSS metric;
  PRINT("ANN Error on the testset:" 
        << metric.Evaluate(ann,testset.inputs_,testset.labels_));
  

  DEPEND_INFO( );

  return 0;
}

