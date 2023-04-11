/**
 * @file kernelridge_impl.h
 * @author Ozgur Taylan Turan
 *
 * Simple Kernelized Ridge Regression & Kernel Regression(Smoothing) &
 * Semi-Parametric Kernel Ridge Regression
 *
 */
#ifndef KERNELRIDGE_IMPL_H
#define KERNELRIDGE_IMPL_H

//// mlpack
//#include <mlpack/core.hpp>
//// local
//#include "kernelridge.h"
//#include "utils/covmat.h"

namespace algo {
namespace regression {

///////////////////////////////////////////////////////////////////////////////
// Kernel Ridge Regression
///////////////////////////////////////////////////////////////////////////////

template<class T>
template<class... Ts>
KernelRidge<T>::KernelRidge ( const arma::mat& inputs,
                              const arma::rowvec& labels,
                              const double& lambda,
                              const Ts&... args ) :
    cov_(args...), lambda_(lambda)
{
  Train(inputs, labels);
}


template<class T>
void KernelRidge<T>::Train ( const arma::mat& inputs,
                             const arma::rowvec& labels )
{
  train_inp_ = inputs.t();
  
  arma::mat k_xx = cov_.GetMatrix(train_inp_,train_inp_);

  arma::mat KLambda = k_xx+
       (lambda_ + 1.e-6) * arma::eye<arma::mat>(k_xx.n_rows, k_xx.n_rows);

  parameters_ = arma::solve(KLambda, labels.t());
}

template<class T>
void KernelRidge<T>::Predict ( const arma::mat& inputs,
                               arma::rowvec& labels ) const
{
  arma::mat k_xpx = cov_.GetMatrix(inputs.t(),train_inp_);
  labels = (k_xpx * parameters_).t();
}

template<class T>
double KernelRidge<T>::ComputeError ( const arma::mat& inputs,
                                      const arma::rowvec& labels ) const
{
  arma::rowvec temp;
  Predict(inputs, temp);
  const size_t n_points = inputs.n_cols;

  temp = labels - temp;

  const double cost = arma::dot(temp, temp) / n_points;

  return cost;
}

///////////////////////////////////////////////////////////////////////////////
// Kernel Regression
///////////////////////////////////////////////////////////////////////////////
template<class T>
template<class... Ts>
Kernel<T>::Kernel ( const arma::mat& inputs,
                    const arma::rowvec& labels,
                    const Ts&... args ) : cov_(args...)
{
  Train(inputs, labels);
}


template<class T>
void Kernel<T>::Train ( const arma::mat& inputs,
                        const arma::rowvec& labels )
{
  train_inp_ = inputs.t();
  train_lab_ = labels;
}

template<class T>
void Kernel<T>::Predict ( const arma::mat& inputs,
                          arma::rowvec& labels ) const
{
  arma::mat sim = cov_.GetMatrix(train_inp_, inputs.t());
  const size_t N = sim.n_rows;
  for(size_t i=0; i<N; i++)
  {
    sim.row(i) /= arma::sum(sim,0);  
  }
  labels = train_lab_ * sim;
}

template<class T>
double Kernel<T>::ComputeError ( const arma::mat& inputs,
                                 const arma::rowvec& labels ) const
{
  arma::rowvec temp;
  Predict(inputs, temp);
  const size_t n_points = inputs.n_cols;

  temp = labels - temp;

  const double cost = arma::dot(temp, temp) / n_points;

  return cost;
}

template<class T, class F>
template<class... Ts>
SemiParamKernelRidge<T,F>::SemiParamKernelRidge ( const arma::mat& inputs,
                                                  const arma::rowvec& labels,
                                                  const double& lambda,
                                                  const size_t& num_funcs,
                                                  const Ts&... args ) :
    cov_(args...), lambda_(lambda), M_(num_funcs), func_(num_funcs)
{
  Train(inputs, labels);
}


template<class T, class F>
void SemiParamKernelRidge<T,F>::Train ( const arma::mat& inputs,
                                        const arma::rowvec& labels )
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
void SemiParamKernelRidge<T,F>::Predict ( const arma::mat& inputs,
                                          arma::rowvec& labels ) const
{
  arma::mat K = cov_.GetMatrix(inputs.t(),train_inp_);
  arma::mat psi = func_.Predict(inputs,"Phase");
  arma::mat A = arma::join_rows(K, psi.t()); 
  labels = (A * parameters_).t();
}

template<class T, class F>
double SemiParamKernelRidge<T,F>::ComputeError ( const arma::mat& inputs,
                                                 const arma::rowvec& labels )
                                                                         const
{
  arma::rowvec temp;
  Predict(inputs, temp);
  const size_t n_points = inputs.n_cols;

  temp = labels - temp;

  const double cost = arma::dot(temp, temp) / n_points;

  return cost;
}
} // namespace regression
} // namespace algo
#endif 
