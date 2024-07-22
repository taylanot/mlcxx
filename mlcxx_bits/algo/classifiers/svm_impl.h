/**
 * @file svm.h
 * @author Ozgur Taylan Turan
 *
 * SVM Classfier
 *
 */

#ifndef SVM_IMPL_H
#define SVM_IMPL_H

namespace algo {
namespace classification {
 
template<class KERNEL,class T>
template<class... Args>
KernelSVM<KERNEL,T>::KernelSVM ( const arma::Mat<T>& inputs,
                                 const arma::Row<size_t>& labels,
                                 const double& C,
                                 const Args&... args) : C_(C), cov_(args...)
{
  Train(inputs,labels);
}

template<class KERNEL,class T>
template<class... Args>
KernelSVM<KERNEL,T>::KernelSVM ( const arma::Mat<T>& inputs,
                                 const arma::Row<size_t>& labels,
                                 const Args&... args ): C_(1e-5), cov_(args...)
{
  Train(inputs,labels);
}

template<class KERNEL,class T>
void KernelSVM<KERNEL,T>::Train ( const arma::Mat<T>& X,
                                  const arma::Row<size_t>& y)
{
  ulab_ = arma::unique(y);

  BOOST_ASSERT_MSG(ulab_.n_elem == 2, "KernelSVM:Only binary labels, please!");

  X_ = &X;

  y_  = (arma::conv_to<arma::Row<int>>::from((y==ulab_(0)) * -2 + 1));

  int n_samples = X.n_cols;
  alphas_.ones(n_samples);
  
  // Compute the kernel matrix
  
  auto K = cov_.GetMatrix(X,X);
  // Formulate the QP problem
  arma::Mat<T> P = K % ((y_).t() * (y_));

  arma::Row<T> q = -arma::ones<arma::Row<T>>(n_samples);
  arma::Mat<T> G = arma::join_cols(
                              -arma::eye<arma::Mat<T>>(n_samples, n_samples),
                              arma::eye<arma::Mat<T>>(n_samples, n_samples));
  arma::Row<T> h = arma::join_rows(arma::zeros<arma::Row<T>>(n_samples),
                                  arma::ones<arma::Row<T>>(n_samples) * C_);

  arma::Mat<T> A = arma::conv_to<arma::Mat<T>>::from(y_);
  arma::Row<T> b = arma::zeros<arma::Row<T>>(1);
  bool success = opt::quadprog(alphas_,P, q, G, h, A, b);
  if (!success)
    PRINT_ERR("HiGHS did not terminate by itself!")

  idx_ = arma::find(alphas_ > 1e-4); // Find the support vectors
  b_ = arma::accu(arma::conv_to<arma::Row<T>>::from(y_.cols(idx_))
        - ((alphas_ % y_) * cov_.GetMatrix(X, X.cols(idx_))))/idx_.n_elem;
}

template<class KERNEL,class T>
void KernelSVM<KERNEL,T>::Classify ( const arma::Mat<T>& inputs,
                                     arma::Row<size_t>& labels ) const
{
  if (idx_.n_elem>0)
  {
    labels.set_size(inputs.n_cols);
    arma::Row<T> temp = alphas_.cols(idx_) % y_.cols(idx_) *
                                    cov_.GetMatrix(X_->cols(idx_), inputs) + b_;
    labels.elem( arma::find( temp <= 0.) ).fill(ulab_[0]);
    labels.elem( arma::find( temp > 0.) ).fill(ulab_[1]);
  }
  else
    PRINT_ERR("No support vectors->No prediction")
      return;
}

template<class KERNEL,class T>
T KernelSVM<KERNEL,T>::ComputeError ( const arma::Mat<T>& points, 
                                      const arma::Row<size_t>& responses ) const
{
  arma::Row<size_t> predictions;
  Classify(points,predictions);
  arma::Row<size_t> temp =  predictions - responses; 
  return (arma::accu(temp != 0))/T(predictions.n_elem);
}

template<class KERNEL,class T>
T KernelSVM<KERNEL,T>::ComputeAccuracy ( const arma::Mat<T>& points, 
                                         const arma::Row<size_t>& responses )
const
{
  return (1. - ComputeError(points, responses))*100;
}

} // namespace classification
} // namespace algo
#endif
