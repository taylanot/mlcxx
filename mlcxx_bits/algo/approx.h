/**
 * @file approx.h
 * @author Ozgur Taylan Turan
 *
 * Function Approximations
 *
 */

#ifndef APPROX_H
#define APPROX_H

namespace algo { 
namespace approx {
  
class Taylor
{
  public:

  /**
   * Non-working model 
   */
  Taylor ( ) : order_(size_t(1)), h_(double(1.)), type_("backward"){ };

  /**
   * @param order : self explanatory
   */
  Taylor ( const size_t& order, const double& h, const std::string type )
    : order_(order), h_(h), type_(type)
  { } ;

  /**
   * @param around : x0
   */
  template<class FUNC>
  Taylor ( FUNC& f, const arma::mat& x0 ) {Train(f,x0);};

  /**
   * @param around : x0
   */
  template<class FUNC>
  void Train (  FUNC& f, const arma::mat& x0 );

  /**
   * @param inputs  : X*
   * @param labels : y*
   */
  void Predict ( const arma::mat& inputs,
                 arma::rowvec& labels ) const;

  /**
   * Calculate the Error Rate
   *
   * @param inputs  : X*
   * @param labels  : y* 
   */
  double ComputeError ( const arma::mat& x , 
                        const arma::rowvec& y ) const;
  /**
   * Calculate the Accuracy
   *
   * @param inputs  : X*
   * @param labels  : y*
   * 
   */
  arma::mat& Parameters() { return param_; }


  /**
   * Serialize the model.
   */
  template<typename Archive>
  void serialize ( Archive& ar,
                   const unsigned int /* version */ )
  {
    ar & BOOST_SERIALIZATION_NVP(order_);
    ar & BOOST_SERIALIZATION_NVP(h_);
    ar & BOOST_SERIALIZATION_NVP(type_);
  }

  private:

  size_t order_;
  double h_;
  std::string type_;

  arma::rowvec param_;
  arma::mat x0_;
  
  
}; 
} // namespace approx
} // namespace algo

#include "approx_impl.h"

#endif


