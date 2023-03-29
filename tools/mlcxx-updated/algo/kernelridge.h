/**
 * @file kernelridge.h
 * @author Ozgur Taylan Turan
 *
 * Simple Kernelized Ridge Regression & Kernel Regression(Smoothing)
 *
 * TODO: Weighted regression example
 */

#ifndef KERNELRIDGE_H
#define KERNELRIDGE_H

//#include <mlpack/prereqs.hpp>

namespace algo { 
namespace regression {

///////////////////////////////////////////////////////////////////////////////
// Kernel Ridge Regression
///////////////////////////////////////////////////////////////////////////////

template<class T>
class KernelRidge 
{
  public:

  /**
   * @param inputs X
   * @param labels y
   * @param lambda regularization hyper-parameter
   * @param args for the kernel
   */
  template<typename... Ts>
  KernelRidge<T> ( const arma::mat& inputs,
                   const arma::rowvec& labels,
                   const double lambda,
                   const Ts&... args );
  /**
   * Non-working model 
   */
  template<typename... Ts>
  KernelRidge<T> ( const Ts&... args ) : cov_(args...), lambda_(0.0) { }

  /**
   * @param inputs X
   * @param labels y
   */
  void Train ( const arma::mat& inputs,
               const arma::rowvec& labels );

    /**
   * @param inputs X*
   * @param labels y*
   */
  void Predict ( const arma::mat& inputs,
                 arma::rowvec& labels ) const;

  /**
   * Calculate the L2 squared error on the given predictors and responses using
   * this linear regression model. This calculation returns
   *
   * @param inputs 
   * @param labels 
   */
  double ComputeError ( const arma::mat& points,
                        const arma::rowvec& responses ) const;

  const arma::vec& Parameters ( ) const { return parameters_; }

  arma::vec& Parameters ( ) { return parameters_; }

  double Lambda ( ) const { return lambda_; }

  double& Lambda ( ) { return lambda_; }

  /**
   * Serialize the model.
   */
  template<typename Archive>
  void serialize ( Archive& ar, const unsigned int /* version */ )
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

///////////////////////////////////////////////////////////////////////////////
// Kernel Regression
///////////////////////////////////////////////////////////////////////////////

template<class T>
class Kernel
{
  public:

  /**
   * @param inputs X
   * @param labels y
   * @param args for the kernel
   */
  template<typename... Ts>
  Kernel<T> ( const arma::mat& inputs,
              const arma::rowvec& labels,
              const Ts&... args );

  /**
   * Non-working model 
   */
  template<typename... Ts>
  Kernel<T> ( const Ts&... args ) : cov_(args...){ }

  /**
   * @param inputs X
   * @param labels y
   * @param args for the kernel
   */
  void Train ( const arma::mat& inputs,
               const arma::rowvec& labels );

  /**
   * @param inputs X*
   * @param labels y*
   */
  void Predict ( const arma::mat& inputs,
                 arma::rowvec& labels ) const;

  /**
   * Calculate the L2 squared error on the given predictors and responses using
   * this linear regression model. This calculation returns
   *
   * @param inputs 
   * @param labels 
   */
  double ComputeError ( const arma::mat& points,
                        const arma::rowvec& responses) const;

  const arma::mat& Parameters ( ) const { return parameters_; }

  arma::mat& Parameters ( ) { return parameters_; }


  /**
   * Serialize the model.
   */
  template<typename Archive>
  void serialize ( Archive& ar, const unsigned int /* version */ )
  {
    ar & BOOST_SERIALIZATION_NVP(parameters_);
    ar & BOOST_SERIALIZATION_NVP(train_inp_);
    ar & BOOST_SERIALIZATION_NVP(train_lab_);
  }

  private:
  arma::mat train_inp_;
  arma::rowvec train_lab_;
  arma::vec parameters_;
  utils::covmat<T> cov_;
};

} // namespace regression
} // namespace algo

#include "kernelridge_impl.h"

#endif

