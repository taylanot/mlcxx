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
template<class KERNEL=mlpack::GaussianKernel>
class Parzen
{
  public:

  /**
   * Non-working model 
   */
  Parzen ( ) = default;

  /**
   * @param h : window
   */
  Parzen ( const double& h ) : h_(h) { } ;

  /**
   * @param inputs    : X
   * @param labels    : y
   * @param num_class : number of classes
   * @param h         : window
   */
  Parzen ( const arma::mat& inputs,
           const arma::Row<size_t>& labels,
           const size_t& num_class,
           const double& h );
  /**
   * @param inputs  : X
   * @param labels  : y
   * @param num_class : number of classes
   */
  Parzen ( const arma::mat& inputs,
           const arma::Row<size_t>& labels,
           const size_t& num_class );

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

  double h_ = 1.e-5;
  
  arma::mat parameters_;

  arma::Row<size_t> unique_;
  /**
   * @param inputs  : X
   * @param labels  : y
   */
  void Density ( const arma::mat& inputs,
                 const arma::Row<size_t>& labels );
};

//=============================================================================
// Nearest Neighbour Classifier 
// * You can do this faster with search of mlpack maybe...
//=============================================================================
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
    ar & BOOST_SERIALIZATION_NVP(dim_);
    ar & BOOST_SERIALIZATION_NVP(nuclass_);
    ar & BOOST_SERIALIZATION_NVP(nclass_);
    ar & BOOST_SERIALIZATION_NVP(size_);
    ar & BOOST_SERIALIZATION_NVP(k_);
    ar & BOOST_SERIALIZATION_NVP(inputs_);
    ar & BOOST_SERIALIZATION_NVP(labels_);

  }

  private:

  size_t k_;
  size_t dim_;
  size_t size_;
  size_t nclass_;
  size_t nuclass_;

  arma::Row<size_t> unique_;

  
  arma::Mat<T> inputs_;
  arma::Row<size_t> labels_;

};

} // namespace classification
} // namespace algo

#include "nonparamclass_impl.h"

#endif

