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

///////////////////////////////////////////////////////////////////////////////
// Gaussian Process Regression
///////////////////////////////////////////////////////////////////////////////

template<class T>
class GP
{
  public:
  /**
   * Non-working model 
   * @param args  : kernel parameters
   */
  template<typename... Ts>
  GP<T> ( const Ts&... args ) : cov_(args...), lambda_(0.0) { };

  /**
   * @param X     : inputs 
   * @param y     : labels 
   * @param args  : kernel parameters 
   */
  template<typename... Ts>
  GP<T> ( const arma::mat& inputs,
          const arma::rowvec& labels,
          const double& lambda,
          const Ts&... args );
  
  /**
   * @param X     : inputs 
   * @param y     : labels 
   */
  void Train ( const arma::mat& inputs,
               const arma::rowvec& labels );

  /**
   * @param X*     : inputs*
   * @param y*     : labels* 
   */
  void Predict ( const arma::mat& inputs,
                 arma::rowvec& labels ) const;

  /**
   * @param X*     : inputs*
   * @param y*     : labels* 
   */
  void PredictVariance ( const arma::mat& inputs,
                         arma::rowvec& labels ) const;

  /**
   * @param X*     : inputs*
   * @param y*     : labels* 
   */
  void PredictVariance ( const arma::mat& inputs,
                         arma::mat& labels ) const;

  /**
   * Calculate the L2 squared error on the given predictors and responses using
   * this linear regression model. This calculation returns
   *
   * @param X*     : points*
   * @param y*     : responses* 
   */
  double ComputeError ( const arma::mat& inputs,
                        const arma::rowvec& labels ) const;
  /**
   * Calculate the Log Likelihood of the model given data
   *
   * @param X*     : points*
   * @param y*     : responses* 
   */
  double LogLikelihood ( const arma::mat& inputs,
                         const arma::rowvec& labels ) const;
  /**
   * Sample functions from the prior 
   *
   * @param X      : inputs
   * @param y      : labels
   */
  void  SamplePrior ( const size_t& k,
                      const arma::mat& inputs,
                      arma::mat& labels ) const;

  /**
   * Sample functions from the prior 
   *
   * @param X      : inputs
   * @param y      : labels
   */
  void  SamplePosterior ( const size_t& k,
                          const arma::mat& inputs,
                          arma::mat& labels ) const;

  const arma::vec& Parameters ( ) const { return parameters_; }

  arma::vec& Parameters ( ) { return parameters_; }


  /**
   * Serialize the model.
   */
  template<typename Archive>
  void serialize ( Archive& ar, const unsigned int /* version */ )
  {
    ar & BOOST_SERIALIZATION_NVP(parameters_);
    ar & BOOST_SERIALIZATION_NVP(train_inp_);
    ar & BOOST_SERIALIZATION_NVP(lambda_);
    ar & BOOST_SERIALIZATION_NVP(cov_);
    ar & BOOST_SERIALIZATION_NVP(L_);
  }

  private:
  arma::mat cov_train_;   // for check 
  arma::mat cov_predict_; // for check
  arma::mat L_; // for check

  arma::mat train_inp_;   // for later usage
  size_t N_;   // for later usage
  arma::vec parameters_;
  utils::covmat<T> cov_;
  double  lambda_;
};



} // namespace regression
} // namespace algo

#include "gp_impl.h"

#endif

