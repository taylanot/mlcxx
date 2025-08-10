/**
 * @file kernelridge_impl.h
 * @author Ozgur Taylan Turan
 *
 * Simple Kernelized Ridge Regression & Kernel Regression(Smoothing) &
 * Semi-Parametric Kernel Ridge Regression
 *
 */
#ifndef KERNELRIDGE_IMPL_H
#define KERNELRIDGE_IMPL_H

namespace algo {
namespace regression {

//-----------------------------------------------------------------------------
// Kernel Ridge Regression
//-----------------------------------------------------------------------------
template<class KERNEL, class T>
template<class... Ts>
KernelRidge<KERNEL,T>::KernelRidge ( const arma::Mat<T>& inputs,
                                     const arma::Row<T>& labels,
                                     const double& lambda,
                                     const Ts&... args ) :
    cov_(args...), lambda_(lambda)
{
  Train(inputs, labels);
}
///////////////////////////////////////////////////////////////////////////////
template<class KERNEL, class T>
void KernelRidge<KERNEL,T>::Train ( const arma::Mat<T>& inputs,
                                    const arma::Row<T>& labels )
{
  train_inp_ = inputs;
  
  arma::Mat<T> k_xx = cov_.GetMatrix(train_inp_,train_inp_);

  arma::Mat<T> KLambda = k_xx+
       (lambda_ + 1.e-6) * arma::eye<arma::Mat<T>>(k_xx.n_rows, k_xx.n_rows);

  parameters_ = 
    arma::conv_to<arma::Row<T>>::from(arma::solve(KLambda, labels.t()));
}
///////////////////////////////////////////////////////////////////////////////
template<class KERNEL, class T>
void KernelRidge<KERNEL,T>::Predict ( const arma::Mat<T>& inputs,
                                            arma::Row<T>& labels ) const
{

  arma::Mat<T> k_xpx = cov_.GetMatrix(train_inp_,inputs);
  labels = (parameters_ * k_xpx );
}
///////////////////////////////////////////////////////////////////////////////
template<class KERNEL, class T>
T KernelRidge<KERNEL,T>::ComputeError ( const arma::Mat<T>& inputs,
                                        const arma::Row<T>& labels ) const
{
  arma::Row<T> temp;
  Predict(inputs, temp);
  const size_t n_points = inputs.n_cols;

  temp = labels - temp;

  const T cost = (arma::dot(temp, temp) / n_points);

  return cost;
}

//-----------------------------------------------------------------------------
// Kernel Regression
//-----------------------------------------------------------------------------
template<class KERNEL, class T>
template<class... Ts>
Kernel<KERNEL,T>::Kernel ( const arma::Mat<T>& inputs,
                           const arma::Row<T>& labels,
                           const Ts&... args ) : cov_(args...)
{
  Train(inputs, labels);
}
///////////////////////////////////////////////////////////////////////////////
template<class KERNEL, class T>
void Kernel<KERNEL,T>::Train ( const arma::Mat<T>& inputs,
                               const arma::Row<T>& labels )
{
  train_inp_ = inputs;
  train_lab_ = labels;
}
///////////////////////////////////////////////////////////////////////////////
template<class KERNEL, class T>
void Kernel<KERNEL,T>::Predict ( const arma::Mat<T>& inputs,
                                       arma::Row<T>& labels ) const
{
  arma::Mat<T> sim = cov_.GetMatrix(train_inp_, inputs);
  const size_t N = sim.n_rows;
  for(size_t i=0; i<N; i++)
  {
    sim.row(i) /= arma::sum(sim,0);  
  }
  labels = train_lab_ * sim;
}
///////////////////////////////////////////////////////////////////////////////
template<class KERNEL, class T>
T Kernel<KERNEL,T>::ComputeError ( const arma::Mat<T>& inputs,
                                        const arma::Row<T>& labels ) const
{
  arma::Row<T> temp;
  Predict(inputs, temp);
  const size_t n_points = inputs.n_cols;

  temp = labels - temp;

  const T cost = (arma::dot(temp, temp) / n_points);

  return cost;
}

//-----------------------------------------------------------------------------
// Semi-Parametric Kernel Ridge Regression with Mean addition
//-----------------------------------------------------------------------------
template<class KERNEL,class FUNC,class T>
template<class... Ts>
SemiParamKernelRidge2<KERNEL,FUNC,T>::
SemiParamKernelRidge2 ( const arma::Mat<T>& inputs,
                        const arma::Row<T>& labels,
                        const double& lambda,
                        const size_t& num_funcs,
                        const Ts&... args ) :
    cov_(args...), lambda_(lambda), M_(num_funcs), perc_(0.), func_(num_funcs)
{
  Train(inputs, labels);
}
///////////////////////////////////////////////////////////////////////////////
template<class KERNEL,class FUNC,class T>
template<class... Ts>
SemiParamKernelRidge2<KERNEL,FUNC,T>::
SemiParamKernelRidge2 ( const arma::Mat<T>& inputs,
                        const arma::Row<T>& labels,
                        const double& lambda,
                        const double& perc,
                        const Ts&... args ) :

    cov_(args...), lambda_(lambda), M_(0), perc_(perc), func_(perc)
{
  if (perc > 1)
  {
    M_ = perc;
    perc_ = 0.;
  }
  Train(inputs, labels);
}
///////////////////////////////////////////////////////////////////////////////
template<class KERNEL,class FUNC,class T>
void SemiParamKernelRidge2<KERNEL,FUNC,T>::Train ( const arma::Mat<T>& inputs,
                                                   const arma::Row<T>& labels )
{
  train_inp_ = inputs;
  N_ = train_inp_.n_cols;

  psi_ = func_.Predict(inputs); 

  if (M_ == 0)
    M_ = func_.GetM();

  arma::Mat<T> k_xx = cov_.GetMatrix(train_inp_,train_inp_);

  arma::Mat<T> K = k_xx +
       (1.e-8) * arma::eye<arma::Mat<T>>(k_xx.n_rows, k_xx.n_rows);
  
  arma::Mat<T> A = arma::join_rows(K, psi_.t()); 
  arma::Mat<T> B = arma::join_cols(arma::join_rows(K,
                                arma::zeros<arma::Mat<T>>(N_,M_)),
                                arma::zeros<arma::Mat<T>>(M_,N_+M_)); 

  parameters_ = arma::conv_to<arma::Row<T>>::from(
    arma::solve(A.t() * A + lambda_ * B.t(), A.t() * (labels.t()-
                                                      func_.Mean(inputs).t())));
  
}
///////////////////////////////////////////////////////////////////////////////
template<class KERNEL,class FUNC,class T>
void SemiParamKernelRidge2<KERNEL,FUNC,T>::Predict ( const arma::Mat<T>& inputs,
                                                           arma::Row<T>& labels )
{
  arma::Mat<T> K = cov_.GetMatrix(inputs,train_inp_);
  arma::Mat<T> psi = func_.Predict(inputs);
  arma::Mat<T> A = arma::join_rows(K, psi.t()); 
  labels = (A * parameters_.t()).t() + func_.Mean(inputs);
}
///////////////////////////////////////////////////////////////////////////////
template<class KERNEL,class FUNC,class T>
T SemiParamKernelRidge2<KERNEL,FUNC,T>::ComputeError
                                        ( const arma::Mat<T>& inputs,
                                          const arma::Row<T>& labels )
{
  arma::Row<T> temp;
  Predict(inputs, temp);
  const size_t n_points = inputs.n_cols;

  temp = labels - temp;

  const T cost = (arma::dot(temp, temp) / n_points);

  return cost;
}

//-----------------------------------------------------------------------------
// Semi-Parametric Kernel Ridge Regression
//-----------------------------------------------------------------------------
template<class KERNEL,class FUNC,class T>
template<class... Ts>
SemiParamKernelRidge<KERNEL,FUNC,T>::SemiParamKernelRidge 
                                                  ( const arma::Mat<T>& inputs,
                                                    const arma::Row<T>& labels,
                                                    const double& lambda,
                                                    const size_t& num_funcs,
                                                    const Ts&... args ) :
    cov_(args...), lambda_(lambda), M_(num_funcs), perc_(0.), func_(num_funcs)
{
  Train(inputs, labels);
}
///////////////////////////////////////////////////////////////////////////////
template<class KERNEL,class FUNC,class T>
template<class... Ts>
SemiParamKernelRidge<KERNEL,FUNC,T>::SemiParamKernelRidge
                                                  ( const arma::Mat<T>& inputs,
                                                    const arma::Row<T>& labels,
                                                    const double& lambda,
                                                    const double& perc,
                                                    const Ts&... args ) :
    cov_(args...), lambda_(lambda), M_(0), perc_(perc), func_(perc)
{
  //assert ( (perc_<=1. && perc_>0.) && "CHECK given Percentage");
  if (perc > 1)
  {
    M_ = perc;
    perc_ = 0.;
  }
  Train(inputs, labels);
}
///////////////////////////////////////////////////////////////////////////////
template<class KERNEL,class FUNC,class T>
void SemiParamKernelRidge<KERNEL,FUNC,T>::Train ( const arma::Mat<T>& inputs,
                                                  const arma::Row<T>& labels )
{
  train_inp_ = inputs;
  N_ = train_inp_.n_cols;

  psi_ = func_.Predict(inputs); 

  if (M_ == 0)
    M_ = func_.GetM();

  arma::Mat<T> k_xx = cov_.GetMatrix(train_inp_,train_inp_);

  arma::Mat<T> K = k_xx +
       (1.e-8) * arma::eye<arma::Mat<T>>(k_xx.n_rows, k_xx.n_rows);
  
  arma::Mat<T> A = arma::join_rows(K, psi_.t()); 
  arma::Mat<T> B = arma::join_cols(arma::join_rows(K,
                                          arma::zeros<arma::Mat<T>>(N_,M_)),
                                          arma::zeros<arma::Mat<T>>(M_,N_+M_)); 

  parameters_ = arma::conv_to<arma::Row<T>>::from(
    arma::solve(A.t() * A + lambda_ * B.t(), A.t() * labels.t()));
  
}
///////////////////////////////////////////////////////////////////////////////
template<class KERNEL,class FUNC,class T>
void SemiParamKernelRidge<KERNEL,FUNC,T>::Predict ( const arma::Mat<T>& inputs,
                                                          arma::Row<T>& labels )
{
  arma::Mat<T> K = cov_.GetMatrix(inputs,train_inp_);
  arma::Mat<T> psi = func_.Predict(inputs);
  arma::Mat<T> A = arma::join_rows(K, psi.t()); 
  labels = (A * parameters_.t()).t();
}
///////////////////////////////////////////////////////////////////////////////
template<class KERNEL,class FUNC,class T>
T SemiParamKernelRidge<KERNEL,FUNC,T>::ComputeError
                                                ( const arma::Mat<T>& inputs,
                                                  const arma::Row<T>& labels )
{
  arma::Row<T> temp;
  Predict(inputs, temp);
  const size_t n_points = inputs.n_cols;

  temp = labels - temp;

  const T cost = (arma::dot(temp, temp) / n_points);

  return cost;
}
///////////////////////////////////////////////////////////////////////////////
} // namespace regression
} // namespace algo
#endif 
