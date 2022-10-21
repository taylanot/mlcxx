/**
 * @file kernelridge.h
 * @author Ozgur Taylan Turan
 *
 * Simple Kernelized Ridge Regression
 *
 * TODO: Weighted regression example, bias addition
 */

// standard
//#include <stdexcept>
// mlpack
#include <mlpack/core.hpp>
// local
#include "kernelridge.h"
#include "utils/covmat.h"

using namespace mlpack;
using namespace mlpack::regression;

template<class T>
template<typename Types>
KernelRidge<T>::KernelRidge(double t, Types args)
{
  std::cout << t << std::endl;
}

template<class T>
template<typename... Types>
KernelRidge<T>::KernelRidge(const arma::mat& inputs,
                                   const arma::rowvec& labels,
                                   const double lambda,
                                   const bool bias,
                                   const Types&&... args) :
    KernelRidge<T>(inputs, labels, arma::rowvec(), lambda, bias) { }

template<class T>
template<typename... Types>
KernelRidge<T>::KernelRidge(const arma::mat& inputs,
                                   const arma::rowvec& labels,
                                   const arma::rowvec& weights,
                                   const double lambda,
                                   const bool bias,
                                   const Types&&... args) :
    lambda_(lambda),
    bias_(bias)
{
  Train(inputs, labels, weights, bias);
}


template<class T>
void KernelRidge<T>::Train(const arma::mat& inputs,
                           const arma::rowvec& labels,
                           const arma::rowvec& weights,
                           const bool bias)
{
  this->bias_ = bias;
  if (!bias_)
  {
    train_inp_ = inputs;
    
    utils::covmat<T> k_xx(inputs);
    //std::cout << k_xx.matrix << std::endl;
    //utils::covmat<T> k_xxp(inputs, labels);

    arma::mat KLambda = k_xx.matrix +
         lambda_ * arma::eye<arma::mat>(k_xx.matrix.n_rows, k_xx.matrix.n_rows);

    parameters_ = arma::solve(KLambda, labels.t());
    //std::cout << parameters << std::endl;
  }
  else
  {
    std::cerr << "NOT IMPLEMENTED YET!!!" << std::endl;
  }
}

template<class T>
void KernelRidge<T>::Predict(const arma::mat& points,
                             arma::rowvec& predictions) const
{
  if (!bias_)
  {
    utils::covmat<T> k_xpx(points,train_inp_);
    predictions = (k_xpx.matrix * parameters_).t();
    
  }
  else
  {
    std::cerr << "NOT IMPLEMENTED YET!!!" << std::endl;
  }
}

template<class T>
double KernelRidge<T>::ComputeError(const arma::mat& inputs,
                                    const arma::rowvec& labels) const
{
  arma::rowvec temp;
  Predict(inputs, temp);
  const size_t n_points = inputs.n_rows;

  if (!bias_)
  {
    temp = labels - temp;
  }
  else
  {
    std::cerr << "NOT IMPLEMENTED YET!!!" << std::endl;
  }

  const double cost = arma::dot(temp, temp) / n_points;

  return cost;
}

/** DECLARATIONS for available mlpack kernels...
 *
 *  NOTE: if you add another kernel type you must declare it here otherwise you 
 *  will get an error for undefined reference!
*/
template class KernelRidge<mlpack::kernel::CauchyKernel>;
template class KernelRidge<mlpack::kernel::LinearKernel>;
template class KernelRidge<mlpack::kernel::GaussianKernel>;
template class KernelRidge<mlpack::kernel::SphericalKernel>;
template class KernelRidge<mlpack::kernel::LaplacianKernel>;
template class KernelRidge<mlpack::kernel::TriangularKernel>;
template class KernelRidge<mlpack::kernel::PolynomialKernel>;
template class KernelRidge<mlpack::kernel::EpanechnikovKernel>;
template class KernelRidge<mlpack::kernel::HyperbolicTangentKernel>;
