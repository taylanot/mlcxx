/**
 * @file check.cpp
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
  
  std::filesystem::path dir = EXP_PATH/"01_10_24";
  std::filesystem::create_directories(dir);

  data::regression::Dataset trainset(3, 1000);
  data::regression::Dataset testset(3, 1000);


  arma::irowvec Ns = arma::regspace<arma::irowvec>(1,1,10);

  src::LCurve<mlpack::LinearRegression<>,mlpack::MSE> LC(Ns,10000,
                                                            true,false,true);


  {
    trainset.Generate(std::string("Linear"),0.);
    testset.Generate(std::string("Linear"),0.);

    LC.Split(trainset, testset,0,false);

    LC.test_errors_.save(dir/"peak-linear0.csv", arma::csv_ascii);
  }

  {
    trainset.Generate(std::string("Sine"),0.);
    testset.Generate(std::string("Sine"),0.);

    LC.Split(trainset, testset,0,false);

    LC.test_errors_.save(dir/"peak-sine0.csv", arma::csv_ascii);
  }

  {
    trainset.Generate(std::string("Linear"),1.);
    testset.Generate(std::string("Linear"),1.);

    LC.Split(trainset, testset,0,false);

    LC.test_errors_.save(dir/"peak-linear1.csv", arma::csv_ascii);
  }

  {
    trainset.Generate(std::string("Sine"),1.);
    testset.Generate(std::string("Sine"),1.);

    LC.Split(trainset, testset,0,false);

    LC.test_errors_.save(dir/"peak-sine1.csv", arma::csv_ascii);
  }

  PRINT_TIME(timer.toc());

  return 0;
}



