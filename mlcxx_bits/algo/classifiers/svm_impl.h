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
KernelSVM<KERNEL,T>::KernelSVM ( const arma::Mat<T>& inputs,
                                 const arma::Row<int>& labels,
                                 const double& C ) : C_(C)
{
  Train(inputs,labels);
}

template<class KERNEL,class T>
KernelSVM<KERNEL,T>::KernelSVM ( const arma::Mat<T>& inputs,
                                 const arma::Row<int>& labels ) : C_(1e-5)
{
  Train(inputs,labels);
}

template<class KERNEL,class T>
void KernelSVM<KERNEL,T>::Train ( const arma::Mat<T>& X,
                                  const arma::Row<int>& y)
{
  X_ = &X;
  y_ = &y;
  int n_samples = X.n_cols;
  alphas_.ones(n_samples);
  
  // Compute the kernel matrix
  
  auto K = cov_.GetMatrix(X,X);
  // Formulate the QP problem
  arma::Mat<T> P = K % (y.t() * y);

  arma::Row<T> q = -arma::ones<arma::Row<T>>(n_samples);
  arma::Mat<T> G = arma::join_cols(
                              -arma::eye<arma::Mat<T>>(n_samples, n_samples),
                              arma::eye<arma::Mat<T>>(n_samples, n_samples));
  arma::Row<T> h = arma::join_rows(arma::zeros<arma::Row<T>>(n_samples),
                                  arma::ones<arma::Row<T>>(n_samples) * C_);

  arma::Mat<T> A = arma::conv_to<arma::Mat<T>>::from(y);
  arma::Row<T> b = arma::zeros<arma::Row<T>>(1);
  bool success = opt::quadprog(alphas_,P, q, G, h, A, b);
  if (!success)
  {
    std::cerr << "Failed to solve the QP problem." << std::endl;
    return;
  }

  arma::uvec idx= arma::find(alphas_ > 1e-4); // Find the support vectors
  b_ = arma::accu(arma::conv_to<arma::Row<T>>::from(y.cols(idx))
        - arma::sum((alphas_ % y * cov_.GetMatrix(X, X.cols(idx)))))/idx.n_elem;
}

template<class KERNEL,class T>
void KernelSVM<KERNEL,T>::Classify ( const arma::Mat<T>& inputs,
                                     arma::Row<int>& labels ) const
{
  labels.set_size(inputs.n_cols);
  arma::Row<T> temp = alphas_ % (*y_) *  cov_.GetMatrix(*X_, inputs) + b_;
  labels.elem( arma::find( temp > 0.) ).fill(1);
  labels.elem( arma::find( temp < 0.) ).fill(-1);
}

template<class KERNEL,class T>
T KernelSVM<KERNEL,T>::ComputeError ( const arma::Mat<T>& points, 
                                      const arma::Row<int>& responses ) const
{
  arma::Row<int> predictions;
  Classify(points,predictions);
  arma::Row<int> temp =  predictions - responses; 
  return (arma::accu(temp != 0))/T(predictions.n_elem);
}

template<class KERNEL,class T>
T KernelSVM<KERNEL,T>::ComputeAccuracy ( const arma::Mat<T>& points, 
                                         const arma::Row<int>& responses ) const
{
  return (1. - ComputeError(points, responses))*100;
}

} // namespace classification
} // namespace algo
#endif
