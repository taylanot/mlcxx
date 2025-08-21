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

//-----------------------------------------------------------------------------
// Nearest Neighbour Classifier 
//-----------------------------------------------------------------------------
template<class T=DTYPE>
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
   * @param inputs    : X
   * @param labels    : y
   * @param num_class : number of classes
   * @param r         : radius
   */
  NNC ( const arma::Mat<T>& inputs,
        const arma::Row<size_t>& labels,
        const size_t& num_class,
        const size_t& k );
                               
  /**
   * @param inputs  : X
   * @param labels  : y
   */

  NNC ( const arma::Mat<T>& inputs,
        const arma::Row<size_t>& labels,
        const size_t& num_class );

  /**
   * @param inputs  : X
   * @param labels  : y
   */
  void Train ( const arma::Mat<T>& inputs,
               const arma::Row<size_t>& labels );

  /**
   * @param inputs    : X
   * @param labels    : y
   * @param num_class : number of classes
   */
  void Train ( const arma::Mat<T>& inputs,
               const arma::Row<size_t>& labels,
               const size_t num_class );

  /**
   * @param inputs  : X*
   * @param labels  : y*
   */
  void Classify ( const arma::Mat<T>& inputs,
                  arma::Row<size_t>& labels ) const;
  /**
   * @param inputs  : X*
   * @param labels  : y*
   * @param probs   : probabilties per class
   */
  void Classify ( const arma::Mat<T>& inputs,
                  arma::Row<size_t>& labels,
                  arma::Mat<T>& probs ) const;
  /**
   * Calculate the Error Rate
   *
   * @param inputs  : X*
   * @param labels  : y* 
   */
  T ComputeError ( const arma::Mat<T>& points, 
                   const arma::Row<size_t>& responses ) const;
  /**
   * Calculate the Accuracy
   *
   * @param inputs  : X*
   * @param labels  : y*
   * 
   */
  T ComputeAccuracy ( const arma::Mat<T>& points, 
                      const arma::Row<size_t>& responses ) const;

  /**
   * Serialize the model.
   */
  template<typename Archive>
  void serialize ( Archive& ar,
                   const unsigned int /* version */ )
  {
    ar ( cereal::make_nvp("dim",dim_),
         cereal::make_nvp("nuclass",nuclass_),
         cereal::make_nvp("nclass",nclass_),
         cereal::make_nvp("size",size_),
         cereal::make_nvp("unique",unique_),
         cereal::make_nvp("k",k_),
         cereal::make_nvp("inputs",inputs_),
         cereal::make_nvp("labels",labels_) );

  }

  private:

  size_t k_; // number of neighbours
  size_t dim_; // number of features
  size_t size_; // number of training samples
  size_t nclass_;// number of classes present
  size_t nuclass_; // number of unique classes observed

  arma::Row<size_t> unique_; // unique classes 

  
  arma::Mat<T> inputs_; // inputs 
  arma::Row<size_t> labels_; // labels

};

} // namespace classification
} // namespace algo

#include "nonparamclass_impl.h"

#endif

