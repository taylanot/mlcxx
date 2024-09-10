/**
 * @file main.cpp
 * @author Ozgur Taylan Turan
 *
 * This file is for lcdb++ experiments
 *
 */
#define DTYPE double  
mlpack::RandomSeed(SEED);
#include <headers.h>

template<class model>
void genCurve( size_t id )
{
  data::classification::oml::Dataset dataset(list(id));
  arma::irowvec Ns = arma::regspace<arma::irowvec>(1,1,size_t(trainset.size_*0.9));
  src::LCurve<model,mlpack::Accuracy> lcurve(Ns,repeat,true,false,true);
  lcurve.Split(trainset,testset,1e-6);
}

int main ( int argc, char** argv )
{
  std::filesystem::path path = EXP_PATH/"30_08_24/oml";
  std::filesystem::create_directories(path);
  std::filesystem::current_path(path);

  arma::wall_clock timer;
  timer.tic();
  data::classification::oml::Dataset dataset(3,EXP_PATH);
  PRINT_VAR(dataset.size_);
  PRINT_VAR(dataset.dimension_);
  PRINT_VAR(dataset.num_class_);
  
  data::classification::oml::Dataset trainset,testset;
  data::Split(dataset,trainset,testset,0.2);
  size_t repeat = 10;
  arma::irowvec Ns = arma::regspace<arma::irowvec>(1,1,size_t(trainset.size_*0.9));
  src::LCurve<algo::classification::KernelSVM<mlpack::GaussianKernel>,mlpack::Accuracy> lcurve(Ns,repeat,true,false,true);
  /* src::LCurve<algo::classification::LDC<>,mlpack::Accuracy> lcurve(Ns,repeat,true,false,true); */
  /* src::LCurve<algo::classification::LDC<>,mlpack::Accuracy> lcurve(Ns,repeat,true,false,true); */
  lcurve.Split(trainset,testset,1e-6);
  lcurve.test_errors_.save("ldc_3.csv",arma::csv_ascii);

  PRINT_TIME(timer.toc());

  
  return 0;
}
