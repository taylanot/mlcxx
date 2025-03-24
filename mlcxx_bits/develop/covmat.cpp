/**
 * @file covmat.cpp
 * @author Ozgur Taylan Turan
 *
 * Let's try to make the covariance matrix calculation faster...
 */

#define DTYPE double

#include <headers.h>

using Dataset = data::oml::Dataset<>;
/* using Naive = mlpack::NaiveKernelRule<mlpack::GaussianKernel>; */
using Nystroem = mlpack::NystroemKernelRule<mlpack::GaussianKernel>;

int main()
{
  arma::wall_clock timer;
  Dataset dataset(1489);

  utils::covmat<mlpack::GaussianKernel> cov_;
  auto inp = dataset.inputs_;
  /* inp = {{1,2,3,4,5},{5,10,20,30,50}}; */

  timer.tic();
  auto a = cov_.GetMatrix(inp,inp);
  PRINT_TIME(timer.toc());

  timer.tic();
  auto b = cov_.GetMatrix_approx(inp,inp,900);
  PRINT(arma::size(b))
  PRINT_TIME(timer.toc());

  timer.tic();
  arma::mat c;
  arma::vec eval;
  arma::mat evec;
  Nystroem cov;
  cov.ApplyKernelMatrix(inp,c,eval,evec,100);
  PRINT(arma::size(c))
  PRINT_TIME(timer.toc());

  /* timer.tic(); */
  /* auto c = cov_.GetMatrix_faster(inp,inp); */
  /* PRINT_TIME(timer.toc()); */

  /* timer.tic(); */
  /* auto d = cov_.GetMatrix_blazing(inp,inp); */
  /* PRINT_TIME(timer.toc()); */

  PRINT_VAR(arma::approx_equal(a,b,"absdiff",0.1));
  PRINT_VAR(arma::approx_equal(a,c,"absdiff",0.1));
  /* PRINT_VAR(arma::approx_equal(a,d,"absdiff",0.001)); */

  /* mlpack::GaussianKernel kern; */

  /*   PRINT_VAR(kern.Evaluate(inp.col(0),inp.col(0))); */
  /* /1* for (size_t i=0;i<inp.n_rows;i++) *1/ */
  /* /1* { *1/ */
  /* /1*   arma::Row<DTYPE> vec = inp.row[i]; *1/ */
  /* /1*   PRINT_VAR(kern.Evaluate(vec,vec)); *1/ */
    
  /* /1* } *1/ */

  /* PRINT_VAR(a); */
  /* PRINT_VAR(b); */
  /* PRINT_VAR(c); */
  /* PRINT_VAR(d); */


  return 0;
}

