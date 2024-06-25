/**
 * @file dip.cpp
 * @author Ozgur Taylan Turan
 *
 * Looking at the error distribution of dipping case 
 */

#include <headers.h>


using namespace mlpack;
using namespace mlpack::ann;


//-----------------------------------------------------------------------------
//   main : our main function no need to do anything here...
//-----------------------------------------------------------------------------
int main ( int argc, char** argv )
{
  arma::wall_clock timer;
  timer.tic();
  

  std::string dir = "build/risk_estim/outputs/";
  utils::data::classification::Dataset trainset(1, 10000,2);
  utils::data::classification::Dataset testset(1, 10000,2);
  trainset.Generate("Dipping");
  testset.Generate("Dipping");

  arma::irowvec Ns = arma::regspace<arma::irowvec>(1,1000,9900);
  size_t rep = 10000;
  /* src::classification::LCurve<algo::classification::NMC, mlpack::Accuracy> */ 
                                                                    /* LC(Ns,rep); */
  src::LCurve<algo::classification::NMC, mlpack::Accuracy> LC(Ns, 10);
  LC.Bootstrap(trainset.inputs_, trainset.labels_);

  LC.Generate(trainset, testset);
  LC.test_errors_.save(dir+"dip.csv", arma::csv_ascii);
  LC.StratifiedGenerate(trainset, testset);
  LC.test_errors_.save(dir+"dips.csv", arma::csv_ascii);
  PRINT_TIME(timer.toc());

  return 0; 
}
