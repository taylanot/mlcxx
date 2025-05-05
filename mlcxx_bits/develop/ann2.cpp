/*
 * @file ann2.cpp
 * @author Ozgur Taylan Turan
 *
 */

#include <headers.h>

using DATASET = data::Dataset<arma::Mat<DTYPE>>; 
using LOSS = mlpack::MSE;

int main ( int argc, char** argv )
{
  DATASET dataset(2);

  dataset.Linear(1000);

  typedef mlpack::FFN<mlpack::MeanSquaredError> NetworkType;
  NetworkType network;
  network.Add<mlpack::Linear>(1);

  algo::ANN<NetworkType> model(dataset.inputs_,dataset.labels_,network);
  LOSS loss;
  PRINT_VAR(loss.Evaluate(model,dataset.inputs_,dataset.labels_));


}
