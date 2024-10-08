/**
 * @file paramclass.h
 * @author Ozgur Taylan Turan
 *
 * Parametric Classifiers 
 *
 */

#ifndef PARAMCLASS_H
#define PARAMCLASS_H

namespace algo { 
namespace classification {

//=============================================================================
// Linear Discriminant Classifier
//=============================================================================

class LDC
{
  public:

  /**
   * Non-working model 
   */
  LDC ( ) : lambda_(1.e-5) { };

  /**
   * @param lambda  : regularization
   */
  LDC ( const double& lambda ) : lambda_(lambda) { } ;

  /**
   * @param inputs  : X
   * @param labels  : y
   * @param lambda  : regularization
   */
  LDC ( const arma::mat& inputs,
        const arma::Row<size_t>& labels,
        const double& lambda );
   /**
   * @param inputs  : X
   * @param labels  : y
   * @param lambda  : regularization
   * @param priors  : known priors
   */
  LDC ( const arma::mat& inputs,
        const arma::Row<size_t>& labels,
        const double& lambda,
        const arma::rowvec& priors );
                               
  /**
   * @param inputs  : X
   * @param labels  : y
   */

  LDC ( const arma::mat& inputs,
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
    ar & BOOST_SERIALIZATION_NVP(lambda_);
    ar & BOOST_SERIALIZATION_NVP(means_);
    ar & BOOST_SERIALIZATION_NVP(covs_);
    ar & BOOST_SERIALIZATION_NVP(unique_);
    ar & BOOST_SERIALIZATION_NVP(priors_);

  }

  private:

  size_t dim_;
  size_t num_class_;
  size_t size_;

  double lambda_;

  double jitter_ = 0.;
  
  std::map<size_t, arma::rowvec> means_;
  std::map<size_t, arma::mat> covs_;

  arma::mat cov_;
  arma::mat mean_;

  arma::Row<size_t> unique_;
  arma::rowvec priors_;

};

//=============================================================================
// Quadratic Discriminant Classifier
//=============================================================================

class QDC
{
  public:

  /**
   * Non-working model 
   */
  QDC ( ) : lambda_(1.e-5) { };

  /**
   * @param lambda  : regularization
   */
  QDC ( const double& lambda ) : lambda_(lambda) { } ;

  /**
   * @param inputs  : X
   * @param labels  : y
   * @param lambda  : regularization
   */
  QDC ( const arma::mat& inputs,
        const arma::Row<size_t>& labels,
        const double& lambda );
   /**
   * @param inputs  : X
   * @param labels  : y
   * @param lambda  : regularization
   * @param priors  : known priors
   */
  QDC ( const arma::mat& inputs,
        const arma::Row<size_t>& labels,
        const double& lambda,
        const arma::rowvec& priors );
                               
  /**
   * @param inputs  : X
   * @param labels  : y
   */

  QDC ( const arma::mat& inputs,
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
    ar & BOOST_SERIALIZATION_NVP(lambda_);
    ar & BOOST_SERIALIZATION_NVP(means_);
    ar & BOOST_SERIALIZATION_NVP(covs_);
    ar & BOOST_SERIALIZATION_NVP(icovs_);
    ar & BOOST_SERIALIZATION_NVP(unique_);
    ar & BOOST_SERIALIZATION_NVP(priors_);

  }

  private:

  size_t dim_;
  size_t num_class_;
  size_t size_;

  double lambda_;
  
  double jitter_ = 1.e-12;

  std::map<size_t, arma::rowvec> means_;
  std::map<size_t, arma::mat> covs_;
  std::map<size_t, arma::mat> icovs_;

  arma::Row<size_t> unique_;
  arma::rowvec priors_;

};

//=============================================================================
// Fisher's Linear Discriminant Classifier
//=============================================================================

class FDC
{
  public:

  /**
   * Non-working model 
   */
  FDC ( ) : lambda_(1.e-5) { };

  /**
   * @param lambda
   */
  FDC ( const double& lambda ) : lambda_(lambda) { } ;

  /**
   * @param lambda
   * @param inputs X
   * @param labels y
   */
  FDC ( const arma::mat& inputs,
            const arma::Row<size_t>& labels,
            const double& lambda );
                               
  /**
   * @param inputs X
   * @param labels y
   */

  FDC ( const arma::mat& inputs,
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

  arma::Row<size_t> unique_;

};

//=============================================================================
// Nearest Mean Classifier 
//=============================================================================

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

  // THINGS I DO FOR LOVE! (STUPID COMPILERS) //
  /**
   * @param inputs X
   * @param labels y
   * @param shrink s
   */
  NMC ( const arma::mat& inputs,
        const arma::Row<size_t>& labels,
        const double& shrink );
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
    ar & BOOST_SERIALIZATION_NVP(shrink_);
    ar & BOOST_SERIALIZATION_NVP(size_);
  }

  private:

  size_t dim_;
  size_t size_;
  size_t num_class_;
  double shrink_;
  
  arma::mat centroid_;
  arma::mat parameters_;

  arma::Row<size_t> unique_;

  mlpack::EuclideanDistance metric_;

};
} // namespace classification
} // namespace algo

#include "paramclass_impl.h"

#endif

