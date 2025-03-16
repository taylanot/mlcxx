/**
 * @file newopenml.cpp
 * @author Ozgur Taylan Turan
 *
 * Trying to make the openml dataset for regression
 *
 */
#define DTYPE double

#include <headers.h>

using MODEL = algo::regression::KernelRidge<mlpack::GaussianKernel>;
using OpenML = data::oml::Dataset<DTYPE>;

int main ( int argc, char** argv )
{
  /* std::filesystem::path path = "30_08_24/oml/3"; */
  /* std::filesystem::create_directories(path); */
  /* std::filesystem::current_path(path); */

  arma::wall_clock timer;
  timer.tic();
  data::oml::Dataset<DTYPE> dataset(560);

  data::regression::Transformer<mlpack::data::StandardScaler,
                                OpenML > trans(dataset);

  data::report(dataset);
  /* auto dataset = trans.TransInp(dataset); */
  data::report(dataset);
  /* testset = trans.TransInp(testset); */



  /* PRINT(dataset.size_); */
  /* PRINT(dataset.dimension_); */
  /* xval::KFoldCV<MODEL,mlpack::MSE,arma::Mat<DTYPE>,arma::Row<DTYPE>> */ 
  /*                                   xv(5,dataset.inputs_,dataset.labels_,true); */

  /* xv.Evaluate(0.); */

  PRINT_TIME(timer.toc());

  
  return 0;
}
