/**
 * @file wishart.cpp
 * @author Ozgur Taylan Turan
 *
 * This file is for seeing the eigenvalues of wishart matrix 
 *
 */
#define DTYPE double

#include <headers.h>

int main ( int argc, char** argv )
{
  std::filesystem::path path = EXP_PATH/"06_08_23/wishart";
  std::filesystem::create_directories(path);
  std::filesystem::current_path(path);

  arma::wall_clock timer;
  timer.tic();


  size_t D = 5;
  size_t rep = 100000;


  arma::Col<DTYPE> eigs(rep);

  #pragma omp parallel for
  for (size_t i=0;i<rep;i++)
  {
    data::regression::Dataset dataset(D,D-1);
    dataset.Generate(std::string("Linear"));
    eigs(i) = arma::min(arma::eig_sym(dataset.inputs_*dataset.inputs_.t()));
  }

  eigs.save("eigs.csv",arma::csv_ascii);

  
  PRINT_TIME(timer.toc());
  return 0;
}
