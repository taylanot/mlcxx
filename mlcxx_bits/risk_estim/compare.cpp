/**
 * @file compare.cpp
 * @author Ozgur Taylan Turan
 *
 * Checking some peaking curves
 */

//#define ARMA_DONT_USE_LAPACK
//#define ARMA_DONT_USE_BLAS
//#define ARMA_DONT_USE_ARPACK
//#define ARMA_DONT_USE_OPENMP
#include <headers.h>



//-----------------------------------------------------------------------------
//   main : our main function no need to do anything here...
//-----------------------------------------------------------------------------
int main ( int argc, char** argv )
{
  arma::wall_clock timer;
  timer.tic();


  std::filesystem::path path = "build/risk_estim/outputs/";
  std::filesystem::create_directories(path);
  std::filesystem::current_path(path);
  utils::data::regression::Dataset trainset(1, 1000);
  utils::data::regression::Dataset testset(1, 1000);

  arma::irowvec Ns = arma::regspace<arma::irowvec>(1,1,100);

  src::LCurve<algo::regression::ANN<ens::Adam>, mlpack::MSE> LC(Ns,1000,false);

  trainset.Generate("Sine",1.);
  testset.Generate("Sine",1.);

  arma::Row<size_t> layer_info = {1,5,5,5,1};


  if (std::atoi(argv[1]) == 0)
  {
    LC.Additive(trainset.inputs_,trainset.GetLabels(0),layer_info,1);
    LC.test_errors_.save("additive.csv", arma::csv_ascii);
  }
  else if (std::atoi(argv[1]) == 1)
  {
    LC.Split(trainset,testset,layer_info,1);
    LC.test_errors_.save("split.csv", arma::csv_ascii);
  }
  else if (std::atoi(argv[1]) == 2)
  {
    LC.Bootstrap(trainset.inputs_,trainset.GetLabels(0),layer_info,1);

    LC.test_errors_.save("bootstrap.csv", arma::csv_ascii);
  }


  PRINT_TIME(timer.toc());

  return 0;
}
