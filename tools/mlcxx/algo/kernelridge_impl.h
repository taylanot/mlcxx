/**
 * @file kernelridge_impl.h
 * @author Ozgur Taylan Turan
 *
 * Simple Kernelized Ridge Regression & Kernel Regression(Smoothing)
 *
 * TODO: Weighted regression example, bias addition
 */
#ifndef KERNELRIDGE_IMPL_H
#define KERNELRIDGE_IMPL_H

// mlpack
#include <mlpack/core.hpp>
// local
#include "kernelridge.h"
#include "utils/covmat.h"

using namespace mlpack;
using namespace algo::regression;

///////////////////////////////////////////////////////////////////////////////
// Kernel Ridge Regression
///////////////////////////////////////////////////////////////////////////////

template<class T>
template<class... Ts>
KernelRidge<T>::KernelRidge(const arma::mat& inputs,
                            const arma::rowvec& labels,
                            const double lambda,
                            const Ts&... args) :
    cov_(args...), lambda_(lambda)
{
  Train(inputs, labels);
}


template<class T>
void KernelRidge<T>::Train(const arma::mat& inputs,
                           const arma::rowvec& labels)
{
  train_inp_ = inputs.t();
  
  arma::mat k_xx = cov_.GetMatrix(train_inp_,train_inp_);

  arma::mat KLambda = k_xx+
       (lambda_ + 1.e-6) * arma::eye<arma::mat>(k_xx.n_rows, k_xx.n_rows);

  parameters_ = arma::solve(KLambda, labels.t());
}

template<class T>
void KernelRidge<T>::Predict(const arma::mat& inputs,
                             arma::rowvec& labels) const
{
  arma::mat k_xpx = cov_.GetMatrix(inputs.t(),train_inp_);
  labels = (k_xpx * parameters_).t();
}

template<class T>
double KernelRidge<T>::ComputeError(const arma::mat& inputs,
                                    const arma::rowvec& labels) const
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
Kernel<T>::Kernel(const arma::mat& inputs,
                  const arma::rowvec& labels,
                  const Ts&... args) :
    cov_(args...)
{
  Train(inputs, labels);
}


template<class T>
void Kernel<T>::Train(const arma::mat& inputs,
                      const arma::rowvec& labels)
{
  this -> train_inp_ = inputs.t();
  this -> train_lab_ = labels;
}

template<class T>
void Kernel<T>::Predict(const arma::mat& inputs,
                        arma::rowvec& labels) const
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
double Kernel<T>::ComputeError(const arma::mat& inputs,
                               const arma::rowvec& labels) const
{
  arma::rowvec temp;
  Predict(inputs, temp);
  const size_t n_points = inputs.n_cols;

  temp = labels - temp;

  const double cost = arma::dot(temp, temp) / n_points;

  return cost;
}
#endif 
