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
  /* Dataset data(212); */
  
  /* auto Ns = arma::regspace<arma::Row<size_t>>(10,1,12); */
  /* LCurve<mlpack::LinearRegression<>, */
  /*        Dataset, */
  /*        data::RandomSelect, */
  /*        /1* data::Bootstrap, *1/ */
  /*        /1* data::Additive, *1/ */
  /*        /1* mlpack::MSE> curve(Ns,size_t(2),0.2,true,true); *1/ */
  /*        mlpack::MSE> curve(Ns,size_t(10000),DTYPE(0.2),true,true); */
  /* auto lambdas = arma::linspace<arma::Row<DTYPE>>(0,1,10); */
  /* /1* curve.Generate(data); *1/ */
  /* curve.Generate(data,lambdas); */
  /* /1* PRINT_VAR(curve.GetResults()); *1/ */
  /* /1* curve.test_errors_(0,0) = arma::datum::nan; *1/ */
  /* /1* curve.test_errors_(1,2) = arma::datum::nan; *1/ */
  /* /1* PRINT_VAR(curve.GetResults()); *1/ */
  /* /1* curve.Generate(data,lambdas); *1/ */
  /* /1* /2* curve.Generate(data); *2/ *1/ */
  /* /1* PRINT_VAR(curve.GetResults()); *1/ */


  /* /1* arma::mat A(5, 5, arma::fill::randu); *1/ */

  /* /1* A.fill(arma::datum::nan); *1/ */
  /* /1*  arma::uvec indices = arma::find_nan(A); *1/ */
  /* /1* PRINT(A); *1/ */
  /* /1* PRINT_VAR(indices) *1/ */

  auto loaded = LCurve_::Load(std::string("LCurve.bin"));
  LCurve_ task = std::move(*loaded);
  PRINT(task.GetResults());



  return 0;
}
