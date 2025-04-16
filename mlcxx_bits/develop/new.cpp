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

int main ( ) 
{
  Dataset data(212);
  
  auto Ns = arma::regspace<arma::Row<size_t>>(10,1,12);
  LCurve<mlpack::LinearRegression<>,
         Dataset,
         data::RandomSelect,
         /* data::Bootstrap, */
         /* data::Additive, */
         /* mlpack::MSE> curve(Ns,size_t(2),0.2,true,true); */
         mlpack::MSE> curve(Ns,size_t(2),DTYPE(0.2),true,true);
  auto lambdas = arma::linspace<arma::Row<DTYPE>>(0,1,10);
  /* curve.Generate(data); */
  curve.Generate(data,lambdas);
  PRINT_VAR(curve.GetResults());
  curve.test_errors_(0,0) = arma::datum::nan;
  curve.test_errors_(1,2) = arma::datum::nan;
  PRINT_VAR(curve.GetResults());
  curve.Generate(data,lambdas);
  /* curve.Generate(data); */
  PRINT_VAR(curve.GetResults());


  /* arma::mat A(5, 5, arma::fill::randu); */

  /* A.fill(arma::datum::nan); */
  /*  arma::uvec indices = arma::find_nan(A); */
  /* PRINT(A); */
  /* PRINT_VAR(indices) */

  return 0;
}
