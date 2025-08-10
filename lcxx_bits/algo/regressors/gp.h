/**
 * @file gp.h
 * @author Ozgur Taylan Turan
 *
 * Gaussian Process Regression
 *
 */

#ifndef GP_H
#define GP_H


namespace algo { 
namespace regression {

//-----------------------------------------------------------------------------
// Gaussian Process Regression
//-----------------------------------------------------------------------------
template<class K,class T=DTYPE>
class GaussianProcess
{
  public:

  /**
   * Non-working model 
   * @param args  : kernel parameters
   */
  template<typename... Ts>
  GaussianProcess ( const Ts&... args ) : cov_(args...), lambda_(0.0) { };

  /**
   * @param X     : inputs 
   * @param y     : labels 
   * @param args  : kernel parameters 
   */
  template<typename... Ts>
  GaussianProcess ( const arma::Mat<T>& inputs,
                    const arma::Row<T>& labels,
                    const T lambda,
                    const Ts&... args );
  
  /**
   * @param X     : inputs 
   * @param y     : labels 
   */
  void Train ( const arma::Mat<T>& inputs,
               const arma::Row<T>& labels );

  /**
   * @param X*     : inputs*
   * @param y*     : labels* 
   */
  void Predict ( const arma::Mat<T>& inputs,
                       arma::Row<T>& labels ) const;

  /**
   * @param X*     : inputs*
   * @param y*     : labels* 
   */
  void PredictVariance ( const arma::Mat<T>& inputs,
                               arma::Mat<T>& labels );

  /**
   * Calculate the L2 squared error on the given predictors and responses using
   * this linear regression model. This calculation returns
   *
   * @param X*     : points*
   * @param y*     : responses* 
   */
  T ComputeError ( const arma::Mat<T>& inputs,
                   const arma::Row<T>& labels ) const;
  /**
   * Calculate the Log Likelihood of the model given data
   *
   * @param X*     : points*
   * @param y*     : responses* 
   */
  T LogLikelihood ( const arma::Mat<T>& inputs,
                    const arma::Row<T>& labels ) const;
  /**
   * Sample functions from the prior 
   *
   * @param X      : inputs
   * @param y      : labels
   */
  void  SamplePrior ( const size_t k,
                      const arma::Mat<T>& inputs,
                      arma::Mat<T>& labels ) const;

  /**
   * Sample functions from the prior 
   *
   * @param X      : inputs
   * @param y      : labels
   */
  void  SamplePosterior ( const size_t k,
                          const arma::Mat<T>& inputs,
                          arma::Mat<T>& labels );
  /**
   * Get Parameters
   *
   */
  const arma::Col<T>& Parameters ( ) const { return parameters_; }

  arma::Col<T>& Parameters ( ) { return parameters_; }

  /**
   * Set Lambda
   *
   * @param lambda      : lambda
   */
  void Lambda( const T& lambda ) { lambda_ = lambda; }

  /**
   * Serialize the model.
   */
  template<typename Archive>
  void serialize ( Archive& ar, const unsigned int /* version */ )
  {
    ar ( cereal::make_nvp("parameters",parameters_),
         cereal::make_nvp("train_inp",train_inp_),
         cereal::make_nvp("lambda",lambda_),
         cereal::make_nvp("cov",cov_),
         cereal::make_nvp("L",L_) );
  }

  private:
  arma::Mat<T> cov_train_;   // for check 
  arma::Mat<T> cov_predict_; // for check
  arma::Mat<T> L_; // for check

  arma::Mat<T> train_inp_;   // for later usage
  size_t N_;   // for later usage
  arma::Col<T> parameters_;
  data::Gram<K,T> cov_;
  T lambda_;
};



} // namespace regression
} // namespace algo

#include "gp_impl.h"

#endif

