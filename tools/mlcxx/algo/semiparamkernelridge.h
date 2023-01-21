/**
 * @file semiparamkernelridge.h
 * @author Ozgur Taylan Turan
 *
 * Kernelized Ridge Regression with Semi-parametric Representer Theorem
 *
 */

#ifndef SEMIPARAM_KERNELRIDGE_H
#define SEMIPARAM_KERNELRIDGE_H

#include <mlpack/prereqs.hpp>

namespace algo { 
namespace regression {

template<class T, class F>
class SemiParamKernelRidge 
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
  template<class... Ts>
  SemiParamKernelRidge<T,F>(const arma::mat& inputs,
                          const arma::rowvec& labels,
                          const double& lambda ,
                          const size_t& num_funcs,
                          const Ts&... args );
  /**
   * Non-working model 
   */
  template<typename... Ts>
  SemiParamKernelRidge<T,F>(const Ts&... args) : cov_(args...), lambda_(0.0),
                                  M_(size_t(0)), func_(size_t(0))  { }

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
    ar & BOOST_SERIALIZATION_NVP(M_);
    ar & BOOST_SERIALIZATION_NVP(N_);
    ar & BOOST_SERIALIZATION_NVP(lambda_);
    ar & BOOST_SERIALIZATION_NVP(cov_);
    ar & BOOST_SERIALIZATION_NVP(func_);
    ar & BOOST_SERIALIZATION_NVP(train_inp_);
  }

  private:
  arma::mat cov_train_;   // for check 
  arma::mat cov_predict_; // for check

  arma::mat train_inp_;   // for later usage
  arma::vec parameters_;
  arma::mat psi_;
  utils::covmat<T> cov_;
  double  lambda_;
  size_t M_;
  size_t N_;
  F func_;
};

} // namespace regression
} // namespace mlpack

#include "semiparamkernelridge_impl.h"

#endif

