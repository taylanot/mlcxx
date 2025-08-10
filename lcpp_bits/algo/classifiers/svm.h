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
//-----------------------------------------------------------------------------
// SVM : Kernel version of the support vector classifier
//-----------------------------------------------------------------------------
template<class KERNEL=mlpack::LinearKernel,
         size_t SOLVER=0,
         class T=DTYPE>
class SVM 
{
  public:

  SVM (  ) = default;

  /**
   * Non-working model 
   * @param num_class : number of classes
   * @param args      : kernel parameters
   */
  /* template<class... Args> */
  /* SVM ( const size_t& num_class, */
  /*       const Args&... args ) : C_(T(1.)), cov_(args...), oneclass_(false) */
  /* { }; */

  /**
   * @param num_class : number of classes
   * @param C         : regularization
   * @param args      : kernel parameters
   */
  template<class... Args>
  SVM ( const size_t num_class, const T& C, const Args&... args ) : 
                       solver_("QP"),C_(C),cov_(args...), oneclass_(false) { } ;
  /**
   * @param num_class : number of classes
   * @param solver    : which optimization method QP or SMO
   * @param C         : regularization
   * @param args      : kernel parameters
   */
  template<class... Args>
  SVM ( const size_t num_class, const std::string solver, 
        const T& C, const Args&... args ) :
        solver_(solver),C_(C),cov_(args...), oneclass_(false) { } ;
  /**
   * @param num_class : number of classes
   * @param inputs    : X
   * @param labels    : y
   * @param C         : regularization
   * @param args      : kernel parameters
   */
  template<class... Args>
  SVM ( const arma::Mat<T>& inputs,
        const arma::Row<size_t>& labels,
        const size_t num_class,
        const T& C,
        const Args&... args );
 /**
   * @param num_class : number of classes
   * @param inputs    : X
   * @param labels    : y
   * @param args      : kernel parameters
   */
  template<class... Args>
  SVM ( const arma::Mat<T>& inputs,
        const arma::Row<size_t>& labels,
        const size_t num_class,
        const Args&... args );

  /**
   * @param inputs  : X
   * @param labels  : y
   */
  void Train ( const arma::Mat<T>& inputs,
               const arma::Row<size_t>& labels,
               const size_t num_class );
  /**
   * @param inputs    : X
   * @param labels    : y
   * @param num_class : y
   */
  void Train ( const arma::Mat<T>& inputs,
               const arma::Row<size_t>& labels );

  /**
   * @param inputs  : X*
   * @param labels  : y*
   */
  void Classify ( const arma::Mat<T>& inputs,
                  arma::Row<size_t>& labels );

/**
   * @param inputs    : X*
   * @param labels    : y*
   * @param dec_func  : f(x)
   */
  void Classify ( const arma::Mat<T>& inputs,
                  arma::Row<size_t>& labels,
                  arma::Mat<T>& dec_func );
  /**
   * Calculate the Error Rate
   *
   * @param inputs  : X*
   * @param labels  : y* 
   */
  T ComputeError ( const arma::Mat<T>& points, 
                   const arma::Row<size_t>& responses );
  /**
   * Calculate the Accuracy
   *
   * @param inputs  : X*
   * @param labels  : y*
   * 
   */
  T ComputeAccuracy ( const arma::Mat<T>& points, 
                      const arma::Row<size_t>& responses );

  /**
   * Serialize the model.
   */
  template<typename Archive>
  void serialize ( Archive& ar,
                   const unsigned int /* version */ )
  {
    ar (  cereal::make_nvp("cov",cov_),
          cereal::make_nvp("X",X_),
          cereal::make_nvp("y",y_),
          cereal::make_nvp("alphas",alphas_),
          cereal::make_nvp("ulab",ulab_),
          cereal::make_nvp("idx",idx_),
          cereal::make_nvp("b",b_),
          cereal::make_nvp("C",C_),
          cereal::make_nvp("oneclass",oneclass_),
          cereal::make_nvp("solver",solver_) );
  }

private:
  std::map<int,std::string> solvers_ = {{0,"fanSMO"}};
  std::string solver_ = solvers_[SOLVER];
  size_t nclass_;
  T C_;
  data::Gram<KERNEL> cov_;
  arma::Row<int> y_;
  arma::Mat<T> X_; // When I use here a pointer to the matrix it goes wrong...
  arma::Row<size_t> ulab_;
  arma::Row<T> alphas_;
  arma::Row<T> old_alphas_;
  arma::uvec idx_;
  T b_ = 0;
  bool oneclass_ = false;
  T eps_ = 1e-3;
  T tau_ = 1e-12;
  size_t max_iter_ = 5000;
  size_t iter_ = 0;
  
  OnevAll<SVM<KERNEL,SOLVER,T>> ova_;

  /**
   * @param inputs  : X
   * @param labels  : y
   */

  void _fanSMO ( const arma::Mat<T>& inputs,
                 const arma::Row<size_t>& labels );

  /**
   * @param G : Vector containing the Gradients
   * @param Q : Matrix with the Kernel x y
   */
  std::pair<int, int> _selectset ( arma::Row<T> G, arma::Mat<T> Q );



};

} // namespace classification
} // namespace algo
#include "svm_impl.h"
#endif
