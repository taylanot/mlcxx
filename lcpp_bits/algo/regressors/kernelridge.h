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

//-----------------------------------------------------------------------------
// Kernel Ridge Regression
//-----------------------------------------------------------------------------
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

  template<typename... Ts>
  KernelRidge ( const arma::Mat<T>& inputs,
                const arma::Row<T>& labels );
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
  void serialize ( Archive& ar, const unsigned int )
  {
    ar ( cereal::make_nvp("parameters",parameters_),
         cereal::make_nvp("lambda",lambda_),
         cereal::make_nvp("cov",cov_),
         cereal::make_nvp("train_inp",train_inp_));
  }

  private:

  arma::Mat<T> train_inp_;   // for later usage
  arma::Row<T> parameters_;
  data::Gram<KERNEL> cov_;
  DTYPE lambda_;
};

//-----------------------------------------------------------------------------
// Kernel Regression
//-----------------------------------------------------------------------------
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
    ar (cereal::make_nvp("train_inp",train_inp_),
        cereal::make_nvp("train_lab",train_lab_),
        cereal::make_nvp("cov",cov_));
  }

  private:
  arma::Mat<T> train_inp_;
  arma::Row<T> train_lab_;
  data::Gram<KERNEL> cov_;
};

//-----------------------------------------------------------------------------
// Semi-Parametric Kernel Regression with Mean addition
//-----------------------------------------------------------------------------
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
                          const T lambda ,
                          const size_t num_funcs,
                          const Ts&... args );
  template<class... Ts>
  SemiParamKernelRidge ( const arma::Mat<T>& inputs,
                         const arma::Row<T>& labels,
                         const T lambda ,
                         const T perc,
                         const Ts&... args );
  /**
   * Non-working model 
   */
  template<typename... Ts>
  SemiParamKernelRidge ( const Ts&... args ) : cov_(args...),
                                               lambda_(0.0),
                                               M_(size_t(0)),
                                               perc_(T(0)),
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

  T Lambda() const { return lambda_; }

  T& Lambda() { return lambda_; }

  /**
   * Serialize the model.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */)
  {
    ar (  cereal::make_nvp("parameters",parameters_),
          cereal::make_nvp("M",M_),
          cereal::make_nvp("N",N_),
          cereal::make_nvp("lambda",lambda_),
          cereal::make_nvp("cov",cov_),
          cereal::make_nvp("func",func_),
          cereal::make_nvp("perc",perc_),
          cereal::make_nvp("train_inp",train_inp_) );
  }

  private:

  arma::Mat<T> train_inp_;   // for later usage
  arma::Row<T> parameters_;
  arma::Mat<T> psi_;
  data::Gram<KERNEL> cov_;
  T lambda_;
  size_t M_;
  T perc_;
  size_t N_;
  FUNC func_;

};


} // namespace regression
} // namespace algo

#include "kernelridge_impl.h"

#endif

