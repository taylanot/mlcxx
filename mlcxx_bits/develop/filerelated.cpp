/**
 * @file filerelated.cpp
 * @author Ozgur Taylan Turan
 *
 * This file is for showing double descent mitigation via excluisoin of principle components.
 *
 */
#include "headers.h"

int main ( int argc, char** argv )
{
  std::filesystem::path path = EXP_PATH/"30_08_24/";
  std::filesystem::create_directories(path);
  std::filesystem::current_path(path);

  arma::wall_clock timer;
  timer.tic();

  data::classification::oml::Dataset dataset(3,path);
  PRINT_VAR(dataset.size_);
  PRINT_VAR(dataset.dimension_);
  PRINT_VAR(dataset.num_class_);
  
  data::classification::oml::Dataset trainset,testset;
  data::StratifiedSplit(dataset,trainset,testset,0.2);
  size_t repeat = 10;
  arma::irowvec Ns = arma::regspace<arma::irowvec>(2,1,size_t(trainset.size_*0.9));
  src::LCurve<algo::classification::KernelSVM<>,mlpack::Accuracy> lcurve(Ns,repeat,true,false,true);
  lcurve.Split(trainset,testset);
  lcurve.test_errors_.save("svc.csv",arma::csv_ascii);

  PRINT_TIME(timer.toc());

  
  return 0;
}
