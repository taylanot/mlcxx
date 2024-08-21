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
   * @param args    : kernel parameters
   */
  template<class... Args>
  KernelSVM ( const Args&... args ) : C_(1.), cov_(args...), oneclass_(false)
  { };

  /**
   * @param C : regularization
   * @param args    : kernel parameters
   */
  template<class... Args>
  KernelSVM ( const double& C, const Args&... args ) : C_(C),cov_(args...),
                                                       oneclass_(false) { } ;

  /**
   * @param inputs  : X
   * @param labels  : y
   * @param C       : regularization
   * @param args    : kernel parameters
   */
  template<class... Args>
  KernelSVM ( const arma::Mat<T>& inputs,
              const arma::Row<size_t>& labels,
              const double& C,
              const Args&... args );
 /**
   * @param inputs  : X
   * @param labels  : y
   * @param args    : kernel parameters
   */

  template<class... Args>
  KernelSVM ( const arma::Mat<T>& inputs,
              const arma::Row<size_t>& labels,
              const Args&... args );

  /**
   * @param inputs  : X
   * @param labels  : y
   */
  void Train ( const arma::Mat<T>& inputs,
               const arma::Row<size_t>& labels );

  /**
   * @param inputs  : X*
   * @param labels  : y*
   */
  void Classify ( const arma::Mat<T>& inputs,
                  arma::Row<size_t>& labels ) const;

/**
   * @param inputs    : X*
   * @param labels    : y*
   * @param dec_func  : f(x)
   */
  void Classify ( const arma::Mat<T>& inputs,
                  arma::Row<size_t>& labels,
                  arma::Mat<T>& dec_func ) const;
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
    ar & BOOST_SERIALIZATION_NVP(cov_);
    ar & BOOST_SERIALIZATION_NVP(X_);
    ar & BOOST_SERIALIZATION_NVP(y_);
    ar & BOOST_SERIALIZATION_NVP(alphas_);
    ar & BOOST_SERIALIZATION_NVP(ulab_);
    ar & BOOST_SERIALIZATION_NVP(idx_);
    ar & BOOST_SERIALIZATION_NVP(b_);
    ar & BOOST_SERIALIZATION_NVP(C_);
    ar & BOOST_SERIALIZATION_NVP(oneclass_);
  }

private:
  T C_;
  utils::covmat<KERNEL> cov_;
  const arma::Mat<T>* X_;
  arma::Row<int> y_;
  arma::Row<size_t> ulab_;
  arma::Row<T> alphas_;
  arma::uvec idx_;
  T b_;
  bool oneclass_;


};
} // namespace classification
} // namespace algo
#include "svm_impl.h"
#endif
