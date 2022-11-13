/**
 * @file kernelridge_impl.h
 * @author Ozgur Taylan Turan
 *
 * Simple Kernelized Ridge Regression
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

//template<class T>
//template<class... Ts>
//double KernelRidge<T>::ComputeError(const arma::mat& inputs,
//                                    const arma::rowvec& labels,
//                                    const Ts&... args) const
//{
//  arma::rowvec temp;
//  Predict(inputs, temp, args...);
//  const size_t n_points = inputs.n_cols;
//
//  temp = labels - temp;
//
//  const double cost = arma::dot(temp, temp) / n_points;
//
//  return cost;
//}

#endif 
///** DECLARATIONS for available mlpack kernels...
// *
// *  NOTE: if you add another kernel type you must declare it here otherwise you 
// *  will get an error for undefined reference!
//*/
//template class KernelRidge<mlpack::kernel::CauchyKernel>;
//template class KernelRidge<mlpack::kernel::LinearKernel>;
//template class KernelRidge<mlpack::kernel::GaussianKernel>;
//template class KernelRidge<mlpack::kernel::SphericalKernel>;
//template class KernelRidge<mlpack::kernel::LaplacianKernel>;
//template class KernelRidge<mlpack::kernel::TriangularKernel>;
//template class KernelRidge<mlpack::kernel::PolynomialKernel>;
//template class KernelRidge<mlpack::kernel::EpanechnikovKernel>;
//template class KernelRidge<mlpack::kernel::HyperbolicTangentKernel>;
