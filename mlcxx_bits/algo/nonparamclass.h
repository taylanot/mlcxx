/**
 * @file nonparamclass.h
 * @author Ozgur Taylan Turan
 *
 * Nonarametric Classifiers 
 *
 */

#ifndef NONPARAMCLASS_H
#define NONPARAMCLASS_H

namespace algo { 
namespace classification {

//=============================================================================
// Parzen Classifier
//=============================================================================

class PARZENC
{
  public:

  /**
   * Non-working model 
   */
  PARZENC ( ) : h_(1.e-5) { };

  /**
   * @param h       : window
   */
  PARZENC ( const double& h) : h_(h) { } ;

  /**
   * @param inputs  : X
   * @param labels  : y
   * @param h       : window
   */
  PARZENC ( const arma::mat& inputs,
            const arma::Row<size_t>& labels,
            const double& h );
  /**
   * @param inputs  : X
   * @param labels  : y
   */
  PARZENC ( const arma::mat& inputs,
            const arma::Row<size_t>& labels );

  /**
   * @param inputs  : X
   * @param labels  : y
   */
  void Train ( const arma::mat& inputs,
               const arma::Row<size_t>& labels );

  /**
   * @param inputs  : X*
   * @param labels  : y*
   */
  void Classify ( const arma::mat& inputs,
                  arma::Row<size_t>& labels ) const;

  /**
   * Calculate the Error Rate
   *
   * @param inputs  : X*
   * @param labels  : y* 
   */
  double ComputeError ( const arma::mat& points, 
                        const arma::Row<size_t>& responses ) const;
  /**
   * Calculate the Accuracy
   *
   * @param inputs  : X*
   * @param labels  : y*
   * 
   */
  double ComputeAccuracy ( const arma::mat& points, 
                           const arma::Row<size_t>& responses ) const;
  /**
   * Estimate the prior
   *
   * @param inputs  : X
   * @param labels  : y 
   */
  arma::mat& Parameters() { return parameters_; }


  /**
   * Serialize the model.
   */
  template<typename Archive>
  void serialize ( Archive& ar,
                   const unsigned int /* version */ )
  {
    ar & BOOST_SERIALIZATION_NVP(dim_);
    ar & BOOST_SERIALIZATION_NVP(num_class_);
    ar & BOOST_SERIALIZATION_NVP(size_);
    ar & BOOST_SERIALIZATION_NVP(h_);

  }

  private:

  size_t dim_;
  size_t num_class_;
  size_t size_;

  double h_;
  
  arma::mat parameters_;

  arma::Row<size_t> unique_;

};

//=============================================================================
// Nearest Neighbour Classifier 
// * You can do this faster with search of mlpack maybe...
//=============================================================================

class NNC
{
  public:

  /**
   * Non-working model 
   */
  NNC ( ) : k_(1) { };

  /**
   * @param r  : radius 
   */
  NNC ( const size_t& k ) : k_(k) { } ;

  /**
   * @param inputs  : X
   * @param labels  : y
   * @param r       : radius
   */
  NNC ( const arma::mat& inputs,
        const arma::Row<size_t>& labels,
        const size_t& k );
                               
  /**
   * @param inputs  : X
   * @param labels  : y
   */

  NNC ( const arma::mat& inputs,
        const arma::Row<size_t>& labels );

  /**
   * @param inputs  : X
   * @param labels  : y
   */
  void Train ( const arma::mat& inputs,
               const arma::Row<size_t>& labels );

  /**
   * @param inputs  : X*
   * @param labels  : y*
   */
  void Classify ( const arma::mat& inputs,
                  arma::Row<size_t>& labels ) const;

  /**
   * Calculate the Error Rate
   *
   * @param inputs  : X*
   * @param labels  : y* 
   */
  double ComputeError ( const arma::mat& points, 
                        const arma::Row<size_t>& responses ) const;
  /**
   * Calculate the Accuracy
   *
   * @param inputs  : X*
   * @param labels  : y*
   * 
   */
  double ComputeAccuracy ( const arma::mat& points, 
                           const arma::Row<size_t>& responses ) const;

  /**
   * Serialize the model.
   */
  template<typename Archive>
  void serialize ( Archive& ar,
                   const unsigned int /* version */ )
  {
    ar & BOOST_SERIALIZATION_NVP(dim_);
    ar & BOOST_SERIALIZATION_NVP(num_class_);
    ar & BOOST_SERIALIZATION_NVP(size_);
    ar & BOOST_SERIALIZATION_NVP(k_);
    ar & BOOST_SERIALIZATION_NVP(inputs_);
    ar & BOOST_SERIALIZATION_NVP(labels_);

  }

  private:

  size_t dim_;
  size_t num_class_;
  size_t size_;

  arma::Row<size_t> unique_;

  size_t k_;
  
  arma::mat inputs_;
  arma::Row<size_t> labels_;

};

} // namespace classification
} // namespace algo

#include "nonparamclass_impl.h"

#endif

