/**
 * @file nmc.h
 * @author Ozgur Taylan Turan
 *
 * Nearest Mean Classifier 
 *
 */

#ifndef NMC_H
#define NMC_H

namespace algo { 
namespace classification {

///////////////////////////////////////////////////////////////////////////////
// Nearest Mean Classifier 
///////////////////////////////////////////////////////////////////////////////

class NMC
{
  public:

  /**
   * Non-working model 
   */
  NMC ( );

  /**
   * @param inputs X
   * @param labels y
   */
  NMC ( const arma::mat& inputs,
        const arma::Row<size_t>& labels );
  /**
   * @param inputs X
   * @param labels y
   */
  void Train ( const arma::mat& inputs,
               const arma::Row<size_t>& labels );

  /**
   * @param inputs X*
   * @param labels y*
   */
  void Classify ( const arma::mat& inputs,
                  arma::Row<size_t>& labels ) const;

  /**
   * Calculate the Error Rate
   *
   * @param inputs 
   * @param labels 
   */
  double ComputeError ( const arma::mat& points, 
                        const arma::Row<size_t>& responses ) const;
  /**
   * Calculate the Accuracy
   *
   * @param inputs 
   * @param labels 
   */
  double ComputeAccuracy ( const arma::mat& points, 
                           const arma::Row<size_t>& responses ) const;

  const arma::mat& Parameters() const { return parameters_; }

  arma::mat& Parameters() { return parameters_; }


  /**
   * Serialize the model.
   */
  template<typename Archive>
  void serialize ( Archive& ar,
                   const unsigned int /* version */ )
  {
    ar & BOOST_SERIALIZATION_NVP(parameters_);
    ar & BOOST_SERIALIZATION_NVP(dim_);
    ar & BOOST_SERIALIZATION_NVP(num_class_);
    ar & BOOST_SERIALIZATION_NVP(unique_);
    ar & BOOST_SERIALIZATION_NVP(metric_);
  }

  private:

  size_t dim_;
  size_t num_class_;

  arma::mat parameters_;

  arma::Row<size_t> unique_;

  mlpack::EuclideanDistance metric_;

};

} // namespace classification
} // namespace algo

#include "nmc_impl.h"

#endif

