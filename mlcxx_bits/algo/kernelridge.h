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

template<class T>
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
  KernelRidge<T> ( const arma::mat& inputs,
                   const arma::rowvec& labels,
                   const double& lambda,
                   const Ts&... args );
  /**
   * Non-working model 
   */
  template<typename... Ts>
  KernelRidge<T> ( const Ts&... args ) : cov_(args...), lambda_(0.0) { }

  /**
   * @param inputs X
   * @param labels y
   */
  void Train ( const arma::mat& inputs,
               const arma::rowvec& labels );

  /**
   * @param inputs X*
   * @param labels y*
   */
  void Predict ( const arma::mat& inputs,
                 arma::rowvec& labels ) const;

  /**
   * Calculate the L2 squared error
   *
   * @param inputs 
   * @param labels 
   */
  double ComputeError ( const arma::mat& points,
                        const arma::rowvec& responses ) const;

  const arma::vec& Parameters ( ) const { return parameters_; }

  arma::vec& Parameters ( ) { return parameters_; }

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
  arma::mat cov_train_;   // for check 
  arma::mat cov_predict_; // for check

  arma::mat train_inp_;   // for later usage
  arma::vec parameters_;
  utils::covmat<T> cov_;
  double  lambda_;
};

///////////////////////////////////////////////////////////////////////////////
// Kernel Regression
///////////////////////////////////////////////////////////////////////////////

template<class T>
class Kernel
{
  public:

  /**
   * @param inputs X
   * @param labels y
   * @param args for the kernel
   */
  template<typename... Ts>
  Kernel<T> ( const arma::mat& inputs,
              const arma::rowvec& labels,
              const Ts&... args );

  /**
   * Non-working model 
   */
  template<typename... Ts>
  Kernel<T> ( const Ts&... args ) : cov_(args...){ }

  /**
   * @param inputs X
   * @param labels y
   * @param args for the kernel
   */
  void Train ( const arma::mat& inputs,
               const arma::rowvec& labels );

  /**
   * @param inputs X*
   * @param labels y*
   */
  void Predict ( const arma::mat& inputs,
                 arma::rowvec& labels ) const;

  /**
   * Calculate the L2 squared error
   *
   * @param inputs 
   * @param labels 
   */
  double ComputeError ( const arma::mat& points,
                        const arma::rowvec& responses) const;

  const arma::mat& Parameters ( ) const { return parameters_; }

  arma::mat& Parameters ( ) { return parameters_; }


  /**
   * Serialize the model.
   */
  template<typename Archive>
  void serialize ( Archive& ar, const unsigned int /* version */ )
  {
    ar & BOOST_SERIALIZATION_NVP(parameters_);
    ar & BOOST_SERIALIZATION_NVP(train_inp_);
    ar & BOOST_SERIALIZATION_NVP(train_lab_);
  }

  private:
  arma::mat train_inp_;
  arma::rowvec train_lab_;
  arma::vec parameters_;
  utils::covmat<T> cov_;
};

///////////////////////////////////////////////////////////////////////////////
// Semi-Parametric Kernel Regression with Mean addition
///////////////////////////////////////////////////////////////////////////////

template<class T, class F>
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
  SemiParamKernelRidge2<T,F> ( const arma::mat& inputs,
                               const arma::rowvec& labels,
                               const double& lambda ,
                               const size_t& num_funcs,
                               const Ts&... args );
  template<class... Ts>
  SemiParamKernelRidge2<T,F> ( const arma::mat& inputs,
                               const arma::rowvec& labels,
                               const double& lambda ,
                               const double& perc,
                               const Ts&... args );
  /**
   * Non-working model 
   */
  template<typename... Ts>
  SemiParamKernelRidge2<T,F> ( const Ts&... args ) : cov_(args...),
                                                     lambda_(0.0),
                                                     M_(size_t(0)),
                                                     perc_(double(0)),
                                                     func_(size_t(0))  { }

  /**
   * @param inputs X
   * @param labels y
   */
  void Train ( const arma::mat& inputs,
               const arma::rowvec& labels );
  /**
   * @param inputs X*
   * @param labels y*
   */
  void Predict ( const arma::mat& inputs,
                 arma::rowvec& labels );// const;
  /**
   * Calculate the L2 squared error 
   *
   * @param inputs 
   * @param labels 
   */
  double ComputeError ( const arma::mat& points,
                        const arma::rowvec& responses ); //const;

  const arma::vec& Parameters() const { return parameters_; }

  arma::vec& Parameters() { return parameters_; }

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
  arma::mat cov_train_;   // for check 
  arma::mat cov_predict_; // for check

  arma::mat train_inp_;   // for later usage
  arma::vec parameters_;
  arma::mat psi_;
  utils::covmat<T> cov_;
  double  lambda_;
  size_t M_;
  double perc_;
  size_t N_;
  F func_;

};
///////////////////////////////////////////////////////////////////////////////
// Semi-Parametric Kernel Regression
///////////////////////////////////////////////////////////////////////////////

template<class T, class F>
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
  SemiParamKernelRidge<T,F> ( const arma::mat& inputs,
                              const arma::rowvec& labels,
                              const double& lambda ,
                              const size_t& num_funcs,
                              const Ts&... args );
  template<class... Ts>
  SemiParamKernelRidge<T,F> ( const arma::mat& inputs,
                              const arma::rowvec& labels,
                              const double& lambda ,
                              const double& perc,
                              const Ts&... args );
  /**
   * Non-working model 
   */
  template<typename... Ts>
  SemiParamKernelRidge<T,F> ( const Ts&... args ) : cov_(args...),
                                                    lambda_(0.0),
                                                    M_(size_t(0)),
                                                    perc_(double(0)),
                                                    func_(size_t(0))  { }

  /**
   * @param inputs X
   * @param labels y
   */
  void Train ( const arma::mat& inputs,
               const arma::rowvec& labels );
  /**
   * @param inputs X*
   * @param labels y*
   */
  void Predict ( const arma::mat& inputs,
                 arma::rowvec& labels );// const;
  /**
   * @param inputs X
   * @param labels y
   */
  void _Train ( const arma::mat& inputs,
                const arma::rowvec& labels );
  /**
   * @param inputs X*
   * @param labels y*
   */
  void _Predict ( const arma::mat& inputs,
                  arma::rowvec& labels );//const;


  /**
   * Calculate the L2 squared error 
   *
   * @param inputs 
   * @param labels 
   */
  double ComputeError ( const arma::mat& points,
                        const arma::rowvec& responses ); //const;

  const arma::vec& Parameters() const { return parameters_; }

  arma::vec& Parameters() { return parameters_; }

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
  arma::mat cov_train_;   // for check 
  arma::mat cov_predict_; // for check

  arma::mat train_inp_;   // for later usage
  arma::vec parameters_;
  arma::mat psi_;
  utils::covmat<T> cov_;
  double  lambda_;
  size_t M_;
  double perc_;
  size_t N_;
  F func_;

};
} // namespace regression
} // namespace algo

#include "kernelridge_impl.h"

#endif

