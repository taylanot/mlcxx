/**
 * @file generate_lc.cpp
 * @author Ozgur Taylan Turan
 *
 */

#include <headers.h>

 
//-----------------------------------------------------------------------------
//   main : our main function no need to do anything here...
//-----------------------------------------------------------------------------
int main ( int argc, char** argv )
{

  utils::data::classification::Dataset trainset(2, 6000, 2);
  trainset._dipping(5.,0.1);

  utils::data::classification::Dataset testset(2, 10000, 2);
  testset._dipping(5.,0.1);

  arma::irowvec Ns = arma::regspace<arma::irowvec>(2,1,5000);
  src::classification::LCurve<algo::classification::NMC,
                            mlpack::MSE> lcurve(Ns,1000);

  lcurve.Generate(trainset,testset);
  lcurve.Save("forlcpfn.csv");




  return 0; 
}
