/**
 * @file fisherc.h
 * @author Ozgur Taylan Turan
 *
 * Fisher Classifier
 *
 */

#ifndef FISHERC_H
#define FISHERC_H

namespace algo { 
namespace classification {

///////////////////////////////////////////////////////////////////////////////
// Fisher's Linear Discriminant
///////////////////////////////////////////////////////////////////////////////

class FISHERC
{
  public:

  /**
   * Non-working model 
   */
  FISHERC ( ) : lambda_(1.e-5) { };

  /**
   * @param lambda
   */
  FISHERC ( const double& lambda ) : lambda_(lambda) { } ;

  /**
   * @param lambda
   * @param inputs X
   * @param labels y
   */
  FISHERC ( const double& lambda,
            const arma::mat& inputs,
            const arma::rowvec& labels );
                               
  /**
   * @param inputs X
   * @param labels y
   */

  FISHERC ( const arma::mat& inputs,
            const arma::rowvec& labels );
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
  void Classify ( const arma::mat& inputs,
                  arma::rowvec& labels ) const;

  /**
   * Calculate the Error Rate
   *
   * @param inputs 
   * @param labels 
   */
  double ComputeError ( const arma::mat& points, 
                        const arma::rowvec& responses ) const;
  /**
   * Calculate the Accuracy
   *
   * @param inputs 
   * @param labels 
   */
  double ComputeAccuracy ( const arma::mat& points, 
                           const arma::rowvec& responses ) const;

  const arma::mat& Parameters() const { return parameters_; }

  arma::mat& Parameters() { return parameters_; }


  /**
   * Serialize the model.
   */
  template<typename Archive>
  void serialize ( Archive& ar,
                   const unsigned int /* version */ )
  {
    ar & BOOST_SERIALIZATION_NVP(dim_);
    ar & BOOST_SERIALIZATION_NVP(bias_);
    ar & BOOST_SERIALIZATION_NVP(parameters_);
    ar & BOOST_SERIALIZATION_NVP(num_class_);
    ar & BOOST_SERIALIZATION_NVP(lambda_);

  }

  private:

  size_t dim_;
  size_t num_class_;

  double lambda_;

  arma::mat parameters_;
  double bias_;

  arma::rowvec unique_;

};

} // namespace classification
} // namespace algo

#include "fisherc_impl.h"

#endif

