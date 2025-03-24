/**
 * @file info.cpp
 * @author Ozgur Taylan Turan
 *
 * Investigate the dataset from openml with given id
 *
 */
#define DTYPE double

#include <headers.h>

using OpenML = data::oml::Dataset<DTYPE>;

int main ( int argc, char** argv )
{
  size_t id;
  if (argc == 1)
  {
    ERR("Provide -id...");
    return 0;
  }
  else if (argc > 2)
  {
    ERR("Only provide -id...");
    return 0;
  }
    
  arma::wall_clock timer; timer.tic();

  std::string arg = argv[1];
  id = std::stoi(argv[1]); // Convert string to int

  data::oml::Dataset<DTYPE> dataset(id);

  data::report(dataset);
  mlpack::RandomForest model(dataset.inputs_, dataset.labels_,dataset.num_class_,20,1,1e-7,0,false);

  PRINT_TIME(timer.toc());

  
  return 0;
}
