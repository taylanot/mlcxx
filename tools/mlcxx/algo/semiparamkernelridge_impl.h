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

    cov_(args...), lambda_(lambda), M_(num_funcs), func_(num_funcs)
{
  Train(inputs, labels);
}


template<class T, class F>
void SemiParamKernelRidge<T,F>::Train(const arma::mat& inputs,
                                      const arma::rowvec& labels)
{
  train_inp_ = inputs.t();
  N_ = train_inp_.n_rows;

  psi_ = func_.Predict(inputs, "Phase"); 
  arma::mat k_xx = cov_.GetMatrix(train_inp_,train_inp_);

  arma::mat K = k_xx +
       (1.e-8) * arma::eye<arma::mat>(k_xx.n_rows, k_xx.n_rows);
  
  arma::mat A = arma::join_rows(K, psi_.t()); 
  arma::mat B = arma::join_cols(arma::join_rows(K, arma::zeros(N_,M_)),
                                arma::zeros(M_,N_+M_)); 


  parameters_ = arma::solve(A.t() * A + lambda_ * B.t(), A.t() * labels.t());
  
}

template<class T, class F>
void SemiParamKernelRidge<T,F>::Predict(const arma::mat& inputs,
                             arma::rowvec& labels) const
{
  arma::mat K = cov_.GetMatrix(inputs.t(),train_inp_);
  arma::mat psi = func_.Predict(inputs,"Phase");
  arma::mat A = arma::join_rows(K, psi.t()); 
  labels = (A * parameters_).t();
}


#endif 
