/**
 * @file kernelridge.h
 * @author Ozgur Taylan Turan
 *
 * Simple Kernelized Ridge Regression
 *
 * TODO: Weighted regression example
 */

#ifndef KERNELRIDGE_H
#define KERNELRIDGE_H

#include <mlpack/prereqs.hpp>

namespace algo { 
namespace regression {

template<class T>
class KernelRidge 
{
  public:

  /**
   * @param inputs X
   * @param labels y
   * @param weights 
   * @param lambda regularization hyper-parameter
   * @param bias term
   * @param args for the kernel
   */
  template<typename... Ts>
  KernelRidge<T>(const arma::mat& inputs,
                 const arma::rowvec& labels,
                 const double lambda ,
                 const Ts&... args);
  /**
   * Non-working model 
   */
  template<typename... Ts>
  KernelRidge<T>(const Ts&... args) : cov_(args...), lambda_(0.0) { }

  /**
   * @param inputs X
   * @param labels y
   * @param lambda regularization hyper-parameter
   * @param args for the kernel
   */
  void Train(const arma::mat& inputs,
             const arma::rowvec& labels);
    /**
   * @param inputs X*
   * @param labels y*
   */
  void Predict(const arma::mat& inputs,
               arma::rowvec& labels) const;


  const arma::vec& Parameters() const { return parameters_; }

  arma::vec& Parameters() { return parameters_; }

  double Lambda() const { return lambda_; }

  double& Lambda() { return lambda_; }

  /**
   * Serialize the model.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & BOOST_SERIALIZATION_NVP(parameters_);
    ar & BOOST_SERIALIZATION_NVP(lambda_);
    ar & BOOST_SERIALIZATION_NVP(cov_);
    ar & BOOST_SERIALIZATION_NVP(train_inp_);
  }

  private:
  arma::mat cov_train_;   // for check 
  arma::mat cov_predict_; // for check

  arma::mat train_inp_;   // for later usage
  arma::vec parameters_;
  utils::covmat<T> cov_;
  double  lambda_;
};

} // namespace regression
} // namespace mlpack

#include "kernelridge_impl.h"

#endif

