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

//-----------------------------------------------------------------------------
// Linear Discriminant Classifier
//-----------------------------------------------------------------------------
template<class T=DTYPE>
class LDC
{
  public:

  /**
   * Non-working model 
   */
  LDC ( ) : lambda_(0.) { };

  /**
   * @param lambda  : regularization
   */
  LDC ( const size_t& num_class, const double& lambda ) : num_class_(num_class),
                                                          lambda_(lambda) { } ;

  /**
   * @param inputs    : X
   * @param labels    : y
   * @param num_class : number of classes
   */
  LDC ( const arma::Mat<T>& inputs,
        const arma::Row<size_t>& labels,
        const size_t& num_class );
 
  /**
   * @param inputs    : X
   * @param labels    : y
   * @param num_class : number of classes
   * @param lambda    : regularization
   */
  LDC ( const arma::Mat<T>& inputs,
        const arma::Row<size_t>& labels,
        const size_t& num_class,
        const double& lambda );
   /**
   * @param inputs    : X
   * @param labels    : y
   * @param num_class : number of classes
   * @param lambda    : regularization
   * @param priors    : known priors
   */
  LDC ( const arma::Mat<T>& inputs,
        const arma::Row<size_t>& labels,
        const size_t& num_class,
        const double& lambda,
        const arma::Row<T>& priors );
                               
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
   * @param probs   : scores*
   */
  void Classify ( const arma::Mat<T>& inputs,
                  arma::Row<size_t>& labels,
                  arma::Mat<T>& scores ) const;
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
    ar (cereal::make_nvp("dim",dim_),
        cereal::make_nvp("num_class",num_class_),
        cereal::make_nvp("size",size_),
        cereal::make_nvp("lambda",lambda_),
        cereal::make_nvp("means",means_),
        cereal::make_nvp("covs",covs_),
        cereal::make_nvp("unique",unique_),
        cereal::make_nvp("priors",priors_));

  }

  private:
  public:
  size_t dim_;
  size_t num_class_;
  size_t size_;

  double lambda_;

  double jitter_ = 1.e-8;
  
  // Using maps since it easier to deal with all combinations of labels
  std::map<size_t, arma::Row<T>> means_;
  std::map<size_t, arma::Mat<T>> covs_;

  arma::Mat<T> cov_;
  arma::Mat<T> mean_;

  arma::Row<size_t> unique_;
  arma::Row<size_t> class_;
  arma::Row<T> priors_;

};

//-----------------------------------------------------------------------------
// Quadratic Discriminant Classifier
//-----------------------------------------------------------------------------
template<class T=DTYPE>
class QDC
{
  public:

  /**
   * Non-working model 
   */
  QDC ( ) : lambda_(0.) { };

  /**
   * @param num_class : number of classes
   * @param lambda  : regularization
   */
  QDC ( const size_t& num_class, const double& lambda ) : num_class_(num_class),
                                                          lambda_(lambda) { } ;

  /**
   * @param inputs  : X
   * @param labels  : y
   * @param num_class : number of classes
   * @param lambda  : regularization
   */
  QDC ( const arma::Mat<T>& inputs,
        const arma::Row<size_t>& labels,
        const size_t& num_class,
        const double& lambda );
   /**
   * @param inputs    : X
   * @param labels    : y
   * @param num_class : number of classes
   * @param lambda    : regularization
   * @param priors    : known priors
   */
  QDC ( const arma::Mat<T>& inputs,
        const arma::Row<size_t>& labels,
        const size_t& num_class,
        const double& lambda,
        const arma::Row<T>& priors );
                               
  /**
   * @param inputs    : X
   * @param labels    : y
   * @param num_class : number of classes
   */
  QDC ( const arma::Mat<T>& inputs,
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
   * @param probs   : scores*
   */
  void Classify ( const arma::Mat<T>& inputs,
                  arma::Row<size_t>& labels,
                  arma::Mat<T>& scores ) const;
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
         cereal::make_nvp("num_class",num_class_),
         cereal::make_nvp("size",size_),
         cereal::make_nvp("lambda",lambda_),
         cereal::make_nvp("means",means_),
         cereal::make_nvp("covs",covs_),
         cereal::make_nvp("icovs",icovs_),
         cereal::make_nvp("unique",unique_),
         cereal::make_nvp("priors",priors_));
  }

  private:

  size_t dim_;
  size_t num_class_;
  size_t size_;

  double lambda_;
  
  double jitter_ = 1.e-8;

  std::map<size_t, arma::Row<T>> means_;
  std::map<size_t, arma::Mat<T>> covs_;
  std::map<size_t, arma::Mat<T>> icovs_;

  arma::Row<size_t> unique_;
  arma::Row<size_t> class_;
  arma::Row<T> priors_;

};

//-----------------------------------------------------------------------------
// Nearest Mean Classifier 
//-----------------------------------------------------------------------------
template<class T=DTYPE>
class NMC
{
  public:

  /**
   * Non-working model 
   */
  NMC (  ) { } ;

  /**
   * @param num_class : number of classes
   */
  NMC ( const size_t& num_classes );

  /**
   * @param inputs    : X
   * @param labels    : y
   * @param num_class : number of classes
   */
  NMC ( const arma::Mat<T>& inputs,
        const arma::Row<size_t>& labels,
        const size_t& num_class );

  // THINGS I DO FOR LOVE! (STUPID COMPILERS) //
  /**
   * @param inputs    : X
   * @param labels    : y
   * @param num_class : number of classes
   * @param shrink    : s
   */
  NMC ( const arma::Mat<T>& inputs,
        const arma::Row<size_t>& labels,
        const size_t& num_class,
        const double& shrink );
  /**
   * @param inputs : X
   * @param labels : y
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
   * @param inputs : X*
   * @param labels : y*
   */
  void Classify ( const arma::Mat<T>& inputs,
                  arma::Row<size_t>& labels ) const;

  /**
   * @param inputs  : X*
   * @param labels  : y*
   * @param probs   : scores*
   */
  void Classify ( const arma::Mat<T>& inputs,
                  arma::Row<size_t>& labels,
                  arma::Mat<T>& scores ) const;
  /**
   * Calculate the Error Rate
   *
   * @param inputs : X*
   * @param labels : y
   */
  T ComputeError ( const arma::Mat<T>& points, 
                   const arma::Row<size_t>& responses ) const;
  /**
   * Calculate the Accuracy
   *
   * @param inputs : X* 
   * @param labels : y
   */
  T ComputeAccuracy ( const arma::Mat<T>& points, 
                      const arma::Row<size_t>& responses ) const;

  const arma::Mat<T>& Parameters() const { return parameters_; }

  arma::Mat<T>& Parameters() { return parameters_; }


  /**
   * Serialize the model.
   */
  template<typename Archive>
  void serialize ( Archive& ar,
                   const unsigned int /* version */ )
  {
    ar (  cereal::make_nvp("parameters",parameters_),
          cereal::make_nvp("dim",dim_),
          cereal::make_nvp("num_class",num_class_),
          cereal::make_nvp("unique",unique_),
          cereal::make_nvp("metric",metric_),
          cereal::make_nvp("shrink",shrink_),
          cereal::make_nvp("size",size_) );
  }

  private:

  size_t dim_;
  size_t size_;
  double shrink_;
  size_t num_class_;
  
  arma::Mat<T> centroid_;
  arma::Mat<T> parameters_;

  arma::Row<size_t> unique_;

  mlpack::EuclideanDistance metric_;

};

} // namespace classification
} // namespace algo

#include "paramclass_impl.h"

#endif

