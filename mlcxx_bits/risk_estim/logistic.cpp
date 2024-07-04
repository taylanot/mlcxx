/**
 * @file logistic .cpp
 * @author Ozgur Taylan Turan
 *
 * Looking at the Logistric Regression of mlpack
 */

#include <headers.h>



//-----------------------------------------------------------------------------
//   main : our main function no need to do anything here...
//-----------------------------------------------------------------------------
int main ( int argc, char** argv )
{
  std::filesystem::path path = "build/risk_estim/outputs";
  std::filesystem::create_directories(path);
  std::filesystem::current_path(path);

  arma::wall_clock timer;
  timer.tic();

  utils::data::classification::Dataset dataset(2,10000,2);
  dataset.Generate("Hard");

  /* dataset.Save("data.csv"); */

  arma::irowvec Ns = arma::regspace<arma::irowvec>(1,10,1000);
  /* src::LCurve<mlpack::LogisticRegression<arma::fmat>, utils::ErrorRate> LC(Ns,1000,true); */
  src::LCurve<algo::classification::LDC<DTYPE>, utils::ErrorRate> LC(Ns,1000,true);
  LC.Bootstrap(dataset.inputs_,dataset.labels_);
  LC.test_errors_.save("logit.csv", arma::csv_ascii);

  PRINT_TIME(timer.toc());

  return 0;
}
