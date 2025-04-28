/**
 * @file 01_scale.cpp
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
  
  std::filesystem::path dir = EXP_PATH/"21_11_24";
  std::filesystem::create_directories(dir);

  /* data::oml::Dataset dataset(12); */
  data::classification::Dataset dataset(2,1000,2);
  /* arma::Mat<DTYPE> a(3,1); */
  /* a(0,0) = 1.; */
  /* a(1,0) = 1.; */
  /* a(2,0) = 1.; */
  /* PRINT(arma::cov(a)); */

  dataset.Generate("Simple");

  arma::irowvec Ns = arma::regspace<arma::irowvec>(1,1,20);

  src::LCurve<algo::classification::QDC<>,mlpack::Accuracy> LC(Ns,100);

  
  LC.RandomSet(dataset,2);

  LC.GetResults().save("res.csv",arma::csv_ascii);

  

  /* PRINT_TIME(timer.toc()); */

  return 0;
}



