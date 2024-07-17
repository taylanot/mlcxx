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


int main ( int argc, char** argv )
{
  arma::wall_clock timer;
  timer.tic();

  utils::data::regression::Dataset dataset(1, 1000);
  dataset.Generate("Linear",0.5);

  arma::Row<size_t> bss = {24,32,128};
  arma::Row<double> lrs = {0.0001,0.001}; 
 
  typedef mlpack::FFN<mlpack::MeanSquaredError> NetworkType;
  NetworkType network;
  network.Add<mlpack::Linear>(1);

  algo::ANN<NetworkType> model(&network,lrs[0],bss[0]);

  arma::irowvec Ns = arma::regspace<arma::irowvec>(10,10,100); 
  src::LCurveHPT<algo::ANN<NetworkType>, mlpack::MSE> 
      LCHPT(Ns,10,0.2,false,false,true);

  src::LCurve<algo::ANN<NetworkType>, mlpack::MSE> 
      LC(Ns,10,false,false,true);

  LCHPT.Bootstrap(dataset.inputs_,dataset.GetLabels(0),
      mlpack::Fixed(&network),lrs,bss);
  /* LCHPT.test_errors_.save("lchpt.csv", arma::csv_ascii); */

  LC.Bootstrap(dataset.inputs_,dataset.GetLabels(0),
      &network,lrs[0],bss[0]);

  /* LC.test_errors_.save("lc.csv", arma::csv_ascii); */

  PRINT_TIME(timer.toc())
  return 0;
}




