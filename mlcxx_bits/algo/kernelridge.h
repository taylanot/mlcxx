/**
 * @file kernelridge.h
 * @author Ozgur Taylan Turan
 *
 * Simple Kernelized Ridge Regression & Kernel Regression(Smoothing)
 *
 */

#ifndef KERNELRIDGE_H
#define KERNELRIDGE_H

//#include <mlpack/prereqs.hpp>

namespace algo { 
namespace regression {

///////////////////////////////////////////////////////////////////////////////
// Kernel Ridge Regression
///////////////////////////////////////////////////////////////////////////////

template<class KERNEL, class T=DTYPE>
class KernelRidge 
{
  public:

  /**
   * @param inputs X
   * @param labels y
   * @param lambda regularization hyper-parameter
   * @param args for the kernel
   */
  template<typename... Ts>
  KernelRidge ( const arma::Mat<T>& inputs,
                const arma::Row<T>& labels,
                const double& lambda,
                const Ts&... args );
  /**
   * Non-working model 
   */
  template<typename... Ts>
  KernelRidge ( const Ts&... args ) : cov_(args...), lambda_(0.0) { }

  /**
   * @param inputs X
   * @param labels y
   */
  void Train ( const arma::Mat<T>& inputs,
               const arma::Row<T>& labels );

  /**
   * @param inputs X*
   * @param labels y*
   */
  void Predict ( const arma::Mat<T>& inputs,
                       arma::Row<T>& labels ) const;

  /**
   * Calculate the L2 squared error
   *
   * @param inputs 
   * @param labels 
   */
  T ComputeError ( const arma::Mat<T>& points,
                        const arma::Row<T>& responses ) const;

  const arma::Row<T>& Parameters ( ) const { return parameters_; }

  arma::Row<T>& Parameters ( ) { return parameters_; }

  double Lambda ( ) const { return lambda_; }

  double& Lambda ( ) { return lambda_; }

  /**
   * Serialize the model.
   */
  template<typename Archive>
  void serialize ( Archive& ar, const unsigned int /* version */ )
  {
    ar & BOOST_SERIALIZATION_NVP(parameters_);
    ar & BOOST_SERIALIZATION_NVP(lambda_);
    ar & BOOST_SERIALIZATION_NVP(cov_);
    ar & BOOST_SERIALIZATION_NVP(train_inp_);
  }

  private:
  arma::Mat<T> cov_train_;   // for check 
  arma::Mat<T> cov_predict_; // for check

  arma::Mat<T> train_inp_;   // for later usage
  arma::Row<T> parameters_;
  utils::covmat<KERNEL> cov_;
  double lambda_;
};

///////////////////////////////////////////////////////////////////////////////
// Kernel Regression
///////////////////////////////////////////////////////////////////////////////
template<class KERNEL, class T=DTYPE>
class Kernel
{
  public:

  /**
   * @param inputs X
   * @param labels y
   * @param args for the kernel
   */
  template<typename... Ts>
  Kernel ( const arma::Mat<T>& inputs,
           const arma::Row<T>& labels,
           const Ts&... args );

  /**
   * Non-working model 
   */
  template<typename... Ts>
  Kernel ( const Ts&... args ) : cov_(args...){ }

  /**
   * @param inputs X
   * @param labels y
   * @param args for the kernel
   */
  void Train ( const arma::Mat<T>& inputs,
               const arma::Row<T>& labels );

  /**
   * @param inputs X*
   * @param labels y*
   */
  void Predict ( const arma::Mat<T>& inputs,
                       arma::Row<T>& labels ) const;

  /**
   * Calculate the L2 squared error
   *
   * @param inputs 
   * @param labels 
   */
  T ComputeError ( const arma::Mat<T>& points,
                        const arma::Row<T>& responses) const;

  /**
   * Serialize the model.
   */
  template<typename Archive>
  void serialize ( Archive& ar, const unsigned int /* version */ )
  {
    ar & BOOST_SERIALIZATION_NVP(train_inp_);
    ar & BOOST_SERIALIZATION_NVP(train_lab_);
  }

  private:
  arma::Mat<T> train_inp_;
  arma::Row<T> train_lab_;
  utils::covmat<KERNEL> cov_;
};

///////////////////////////////////////////////////////////////////////////////
// Semi-Parametric Kernel Regression with Mean addition
///////////////////////////////////////////////////////////////////////////////

template<class KERNEL, class FUNC, class T=DTYPE>
class SemiParamKernelRidge2
{
  public:

  /**
   * @param inputs    : X
   * @param labels    : y
   * @param lambda    : regularization hyper-parameter
   * @param num_funcs : number of \psi
   * @param args      : for the kernel hyper-parameters
   */
  template<class... Ts>
  SemiParamKernelRidge2 ( const arma::Mat<T>& inputs,
                          const arma::Row<T>& labels,
                          const double& lambda ,
                          const size_t& num_funcs,
                          const Ts&... args );
  template<class... Ts>
  SemiParamKernelRidge2 ( const arma::Mat<T>& inputs,
                          const arma::Row<T>& labels,
                          const double& lambda ,
                          const double& perc,
                          const Ts&... args );
  /**
   * Non-working model 
   */
  template<typename... Ts>
  SemiParamKernelRidge2 ( const Ts&... args ) : cov_(args...),
                                                lambda_(0.0),
                                                M_(size_t(0)),
                                                perc_(double(0)),
                                                func_(size_t(0))  { }

  /**
   * @param inputs X
   * @param labels y
   */
  void Train ( const arma::Mat<T>& inputs,
               const arma::Row<T>& labels );
  /**
   * @param inputs X*
   * @param labels y*
   */
  void Predict ( const arma::Mat<T>& inputs,
                       arma::Row<T>& labels );// const;
  /**
   * Calculate the L2 squared error 
   *
   * @param inputs 
   * @param labels 
   */
  T ComputeError ( const arma::Mat<T>& points,
                        const arma::Row<T>& responses ); //const;

  const arma::Row<T>& Parameters() const { return parameters_; }

  arma::Row<T>& Parameters() { return parameters_; }

  double Lambda() const { return lambda_; }

  double& Lambda() { return lambda_; }

  /**
   * Serialize the model.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & BOOST_SERIALIZATION_NVP(parameters_);
    ar & BOOST_SERIALIZATION_NVP(M_);
    ar & BOOST_SERIALIZATION_NVP(N_);
    ar & BOOST_SERIALIZATION_NVP(lambda_);
    ar & BOOST_SERIALIZATION_NVP(cov_);
    ar & BOOST_SERIALIZATION_NVP(func_);
    ar & BOOST_SERIALIZATION_NVP(train_inp_);
  }

  private:
  arma::Mat<T> cov_train_;   // for check 
  arma::Mat<T> cov_predict_; // for check

  arma::Mat<T> train_inp_;   // for later usage
  arma::Row<T> parameters_;
  arma::Mat<T> psi_;
  utils::covmat<KERNEL> cov_;
  double  lambda_;
  size_t M_;
  double perc_;
  size_t N_;
  FUNC func_;

};

///////////////////////////////////////////////////////////////////////////////
// Semi-Parametric Kernel Regression
///////////////////////////////////////////////////////////////////////////////
template<class KERNEL, class FUNC, class T=DTYPE>
class SemiParamKernelRidge 
{
  public:

  /**
   * @param inputs    : X
   * @param labels    : y
   * @param lambda    : regularization hyper-parameter
   * @param num_funcs : number of \psi
   * @param args      : for the kernel hyper-parameters
   */
  template<class... Ts>
  SemiParamKernelRidge ( const arma::Mat<T>& inputs,
                         const arma::Row<T>& labels,
                         const double& lambda ,
                         const size_t& num_funcs,
                         const Ts&... args );
  template<class... Ts>
  SemiParamKernelRidge ( const arma::Mat<T>& inputs,
                         const arma::Row<T>& labels,
                         const double& lambda ,
                         const double& perc,
                         const Ts&... args );
  /**
   * Non-working model 
   */
  template<typename... Ts>
  SemiParamKernelRidge ( const Ts&... args ) : cov_(args...),
                                               lambda_(0.0),
                                               M_(size_t(0)),
                                               perc_(double(0)),
                                               func_(size_t(0))  { }

  /**
   * @param inputs X
   * @param labels y
   */
  void Train ( const arma::Mat<T>& inputs,
               const arma::Row<T>& labels );
  /**
   * @param inputs X*
   * @param labels y*
   */
  void Predict ( const arma::Mat<T>& inputs,
                       arma::Row<T>& labels );// const;
  /**
   * Calculate the L2 squared error 
   *
   * @param inputs 
   * @param labels 
   */
  T ComputeError ( const arma::Mat<T>& points,
                        const arma::Row<T>& responses ); //const;

  const arma::Row<T>& Parameters() const { return parameters_; }

  arma::Row<T>& Parameters() { return parameters_; }

  double Lambda() const { return lambda_; }

  double& Lambda() { return lambda_; }

  /**
   * Serialize the model.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & BOOST_SERIALIZATION_NVP(parameters_);
    ar & BOOST_SERIALIZATION_NVP(M_);
    ar & BOOST_SERIALIZATION_NVP(N_);
    ar & BOOST_SERIALIZATION_NVP(lambda_);
    ar & BOOST_SERIALIZATION_NVP(cov_);
    ar & BOOST_SERIALIZATION_NVP(func_);
    ar & BOOST_SERIALIZATION_NVP(train_inp_);
  }

  private:
  arma::Mat<T> cov_train_;   // for check 
  arma::Mat<T> cov_predict_; // for check

  arma::Mat<T> train_inp_;   // for later usage
  arma::Row<T> parameters_;
  arma::Mat<T> psi_;
  utils::covmat<KERNEL> cov_;
  double  lambda_;
  size_t M_;
  double perc_;
  size_t N_;
  FUNC func_;

};
} // namespace regression
} // namespace algo

#include "kernelridge_impl.h"

#endif

