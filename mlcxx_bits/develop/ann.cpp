/*
 * @file ann.cpp
 * @author Ozgur Taylan Turan
 *
 * Ok with ann's there is som much mess due to ensmallen, let me see if I can 
 * identify what is going wrong?
 */

#include <headers.h>



int main ( int argc, char** argv )
{
  data::regression::Dataset dataset(2,10);

  dataset.Generate(std::string("Linear"),0.1);

  size_t rep = 100;

  typedef mlpack::FFN<mlpack::MeanSquaredError> NetworkType;
  
  #pragma omp parallel for
  for (size_t i=0; i<rep; i++)
  {
    NetworkType network;
    network.Add<mlpack::Linear>(1);
    ens::SGD optimizer; 
    network.Train(dataset.inputs_, dataset.labels_, optimizer);
  }

}
