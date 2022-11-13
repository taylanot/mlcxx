/**
 * @file semiparamkernelridge_impl.h
 * @author Ozgur Taylan Turan
 *
 * Kernelized Ridge Regression with Semi-parametric Representer Theorem
 *
 */
#ifndef SEMIPARAMKERNELRIDGE_IMPL_H
#define SEMIPARAMKERNELRIDGE_IMPL_H

// mlpack
#include <mlpack/core.hpp>
// local
#include "semiparamkernelridge.h"
#include "utils/covmat.h"

using namespace algo::regression;

template<class T, class F>
template<class... Ts>
SemiParamKernelRidge<T,F>::SemiParamKernelRidge(const arma::mat& inputs,
                                                const arma::rowvec& labels,
                                                const double& lambda,
                                                const size_t& num_funcs,
                                                const Ts&... args) :

    cov_(args...), lambda_(lambda), num_funcs_(num_funcs), func_(num_funcs)
{
  Train(inputs, labels);
}


template<class T, class F>
void SemiParamKernelRidge<T,F>::Train(const arma::mat& inputs,
                                      const arma::rowvec& labels)
{
  train_inp_ = inputs.t();
  psi_ = func_.Predict(inputs, "Phase"); 
  arma::mat k_xx = cov_.GetMatrix(train_inp_,train_inp_);

  arma::mat KLambda = k_xx+
       (lambda_ + 1.e-6) * arma::eye<arma::mat>(k_xx.n_rows, k_xx.n_rows);

  func_parameters_ = arma::ones<arma::vec>(int(num_funcs_));

  for(int i=0; i<1000; i++)
  {
    data_parameters_ =  arma::solve(KLambda, labels.t()-psi_.t()*func_parameters_);
    func_parameters_ =  arma::solve(psi_.t(), labels.t()-k_xx*data_parameters_);
  }
  
}

template<class T, class F>
void SemiParamKernelRidge<T,F>::Predict(const arma::mat& inputs,
                             arma::rowvec& labels) const
{
  arma::mat k_xpx = cov_.GetMatrix(inputs.t(),train_inp_);
  arma::mat psi = func_.Predict(inputs,"Phase");
  labels = (k_xpx * data_parameters_+psi.t()*func_parameters_).t();
}


#endif 
