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

namespace mlpack { // Add your model to mlpack::regression namespace
namespace regression {

template<class T>
class KernelRidge
{
  public:

  template<typename Types>
  KernelRidge<T>(double t, Types args);


  /**
   * @param inputs X
   * @param labels y
   * @param lambda regularization hyper-parameter
   * @param bias term
   */
  template<typename... Types>
  KernelRidge<T>(const arma::mat& inputs,
                 const arma::rowvec& labels,
                 const double lambda,
                 const bool bias,
                 const Types&&... args);

  /**
   * @param inputs X
   * @param labels y
   * @param weights 
   * @param lambda regularization hyper-parameter
   * @param bias term
   */
  template<typename... Types>
  KernelRidge<T>(const arma::mat& inputs,
                 const arma::rowvec& labels,
                 const arma::rowvec& weights,
                 const double lambda ,
                 const bool bias, 
                 const Types&&... args);
  /**
   * Non-working model 
   */
  KernelRidge<T>() : lambda_(0.0), bias_(false) { }

  /**
   * @param inputs X
   * @param labels y
   * @param lambda regularization hyper-parameter
   * @param bias term
   */
  void Train(const arma::mat& inputs,
             const arma::rowvec& labels,
             const bool bias = false);
  /**
   * @param inputs X
   * @param labels y
   * @param weights 
   * @param lambda regularization hyper-parameter
   * @param bias term
   */
  void Train(const arma::mat& inputs,
             const arma::rowvec& labels,
             const arma::rowvec& weights,
             const bool bias = false);

  /**
   * @param inputs X*
   * @param labels y*
   */
  void Predict(const arma::mat& inputs, arma::rowvec& labels) const;

  /**
   * Calculate MSE Loss 
   * @param inputs X*
   * @param labels y*
   */
  double ComputeError(const arma::mat& inputs,
                      const arma::rowvec& labels) const;

  const arma::vec& Parameters() const { return parameters_; }

  arma::vec& Parameters() { return parameters_; }

  double Lambda() const { return lambda_; }

  double& Lambda() { return lambda_; }

  bool Bias() const { return bias_; }

  /**
   * Serialize the model.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & BOOST_SERIALIZATION_NVP(parameters_);
    ar & BOOST_SERIALIZATION_NVP(lambda_);
    ar & BOOST_SERIALIZATION_NVP(bias_);
  }

  private:
  arma::mat train_inp_;   // for later usage
  arma::vec parameters_;
  arma::mat cov_train_;   // for check 
  arma::mat cov_predict_; // for check

  double  lambda_;
  bool    bias_;

};

} // namespace regression
} // namespace mlpack

#endif // MLPACK_METHODS_LINEAR_REGRESSION_HPP

