/**
 * @file gradientdescent.h
 * @author Ozgur Taylan Turan
 *
 * Simple Gradient Descent Models
 *
 */

#ifndef GRADIENTDESCENT_H
#define GRADIENTDESCENT_H

#include <mlpack/prereqs.hpp>

namespace algo { 
namespace regression {

///////////////////////////////////////////////////////////////////////////////
// LinearRegression
///////////////////////////////////////////////////////////////////////////////

class GDLinear
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
  GDLinear(const arma::mat& inputs,
           const arma::rowvec& labels,
           const bool bias,
           const double lambda ,
           const double lr,
           const size_t t);
  /**
   * Non-working model 
   */
  GDLinear(): lambda_(0.0), bias_(true), lr_(0.01), t_(1000)  { }

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
  /**
   * Calculate the L2 squared error on the given predictors and responses using
   * this linear regression model. This calculation returns
   *
   * @param inputs 
   * @param labels 
   */
  double ComputeError(const arma::mat& points,
                      const arma::rowvec& responses) const;

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
    //ar & BOOST_SERIALIZATION_NVP(bias_);
    //ar & BOOST_SERIALIZATION_NVP(t_);
    //ar & BOOST_SERIALIZATION_NVP(lr_);
  }

  private:
  arma::vec parameters_;
  arma::vec dparameters_;
  double  lambda_;
  bool bias_;
  double lr_;
  size_t t_;
};

}
}

#include "gradientdescent_impl.h"

#endif

