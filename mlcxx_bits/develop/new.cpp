/**
 * @file new.cpp
 * @author Ozgur Taylan Turan
 *
 * I am creating a new learning curve class for better usabaility and cleaner
 * code...
 */

#include <headers.h>


using Dataset = data::oml::Dataset<DTYPE>;
using namespace lcurve;
using LCurve_=LCurve<mlpack::LinearRegression<>,
                    Dataset,
                    data::RandomSelect,
                    /* data::Bootstrap, */
                    /* data::Additive, */
                    mlpack::MSE>;
int main ( ) 
{
  Dataset data(212);
  
  auto Ns = arma::regspace<arma::Row<size_t>>(10,1,15);
  LCurve<mlpack::LinearRegression<>,
         Dataset,
         data::RandomSelect,
         /* data::Bootstrap, */
         /* data::Additive, */
         /* mlpack::MSE> curve(Ns,size_t(2),0.2,true,true); */
         mlpack::MSE> curve(Ns,size_t(10000),DTYPE(0.2),true,true);
  auto lambdas = arma::linspace<arma::Row<DTYPE>>(0,1,10);
  /* curve.Generate(data); */
  curve.Generate(data,lambdas);
  PRINT_VAR(curve.GetResults());

  // For looking at continueation 
  /* curve.test_errors_(0,0) = arma::datum::nan; */
  /* curve.test_errors_(2,1) = arma::datum::nan; */
  /* PRINT_VAR(curve.GetResults()); */
  /* curve.Continue(lambdas); */
  /* PRINT_VAR(curve.GetResults()); */
  /* curve.Continue(lambdas); */
  /* PRINT_VAR(curve.GetResults()); */


  /* arma::mat A(5, 5, arma::fill::randu); */

  /* A.fill(arma::datum::nan); */
  /*  arma::uvec indices = arma::find_nan(A); */
  /* PRINT(A); */
  /* PRINT_VAR(indices) */

  // Continue 
  
  /* auto loaded = LCurve_::Load(std::string("LCurve.bin")); */
  /* LCurve_ task = std::move(*loaded); */
  /* /1* PRINT(task.GetResults()); *1/ */
  /* PRINT_VAR(arma::size(arma::find_nan(task.GetResults()))); */
  /* auto lambdas = arma::linspace<arma::Row<DTYPE>>(0,1,10); */
  /* task.Continue(lambdas); */
  /* /1* PRINT(task.GetResults()); *1/ */
  /* PRINT_VAR(arma::size(arma::find_nan(task.GetResults()))); */
  /* task.Save("LCurve.bin"); */

  // Load Dataset
  /* Dataset data(212); */
  /* PRINT_VAR(data.size_); */
  /* data.Save("dataset.bin"); */
  /* auto loaded = Dataset::Load(std::string("dataset.bin")); */
  /* Dataset data2; */
  /* data2 = std::move(*loaded); */
  /* PRINT_VAR(arma::size(data2.inputs_)); */
  /* PRINT_VAR(arma::size(data2.labels_)); */




  return 0;
}
