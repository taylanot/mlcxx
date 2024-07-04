/**
 * @file svm.h
 * @author Ozgur Taylan Turan
 *
 * SVM Classfier
 *
 */

#ifndef SVM_H
#define SVM_H

namespace algo {
namespace classification {

template<class KERNEL=mlpack::GaussianKernel,
         class T=DTYPE>
class KernelSVM 
{
  public:

  /**
   * Non-working model 
   */
  KernelSVM ( ) : C_(1.e-5) { };

  /**
   * @param C : regularization
   */
  KernelSVM ( const double& C ) : C_(C) { } ;

  /**
   * @param inputs  : X
   * @param labels  : y
   * @param C       : regularization
   */
  KernelSVM ( const arma::Mat<T>& inputs,
              const arma::Row<int>& labels,
              const double& C );
 /**
   * @param inputs  : X
   * @param labels  : y
   */

  KernelSVM ( const arma::Mat<T>& inputs,
              const arma::Row<int>& labels );

  /**
   * @param inputs  : X
   * @param labels  : y
   */
  void Train ( const arma::Mat<T>& inputs,
               const arma::Row<int>& labels );

  /**
   * @param inputs  : X*
   * @param labels  : y*
   */
  void Classify ( const arma::Mat<T>& inputs,
                  arma::Row<int>& labels ) const;

  /**
   * Calculate the Error Rate
   *
   * @param inputs  : X*
   * @param labels  : y* 
   */
  T ComputeError ( const arma::Mat<T>& points, 
                   const arma::Row<int>& responses ) const;
  /**
   * Calculate the Accuracy
   *
   * @param inputs  : X*
   * @param labels  : y*
   * 
   */
  T ComputeAccuracy ( const arma::Mat<T>& points, 
                      const arma::Row<int>& responses ) const;

  /**
   * Serialize the model.
   */
  template<typename Archive>
  void serialize ( Archive& ar,
                   const unsigned int /* version */ )
  {
    ar & BOOST_SERIALIZATION_NVP(cov_);
    ar & BOOST_SERIALIZATION_NVP(X_);
    ar & BOOST_SERIALIZATION_NVP(y_);
    ar & BOOST_SERIALIZATION_NVP(alphas_);
    ar & BOOST_SERIALIZATION_NVP(b_);
    ar & BOOST_SERIALIZATION_NVP(C_);

  }

private:
  utils::covmat<KERNEL> cov_;
  const arma::Mat<T>* X_;
  const arma::Row<int>* y_;
  arma::Row<T> alphas_;
  T b_;
  T C_;


};
} // namespace classification
} // namespace algo
#include "svm_impl.h"
#endif
