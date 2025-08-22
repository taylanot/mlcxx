/**
 * @file gp_impl.h
 * @author Ozgur Taylan Turan
 *
 * Gaussian Process Implementation
 *
 */
#ifndef GP_IMPL_H
#define GP_IMPL_H

namespace algo { 
namespace regression {

//-----------------------------------------------------------------------------
// Gaussian Process Regression
//-----------------------------------------------------------------------------
template<class K, class T>
template<class... Ts>
GaussianProcess<K,T>::GaussianProcess ( const arma::Mat<T>& inputs,
                                        const arma::Row<T>& labels,
                                        const T lambda,
                                        const Ts&... args ) :
                                        cov_(args...), lambda_(lambda)
{
  Train(inputs, labels);
}
///////////////////////////////////////////////////////////////////////////////
template<class K, class T>
GaussianProcess<K,T>::GaussianProcess ( const arma::Mat<T>& inputs,
                                        const arma::Row<T>& labels) :
                                        cov_(), lambda_(0.)
{
  Train(inputs, labels);
}
///////////////////////////////////////////////////////////////////////////////
template<class K, class T>
void GaussianProcess<K,T>::Train ( const arma::Mat<T>& inputs,
                                   const arma::Row<T>& labels )
{
  train_inp_ = inputs;
  N_ = inputs.n_cols;
  
  arma::Mat<T> k_xx = cov_.GetMatrix(train_inp_,train_inp_);

  arma::Mat<T> KLambda = k_xx+
       (lambda_ + 1.e-6) * arma::eye<arma::Mat<T>>(k_xx.n_rows, k_xx.n_rows);

  L_ = arma::chol(KLambda, "lower");
  parameters_ = arma::solve(L_.t(), arma::solve(L_,labels.t()));
}
///////////////////////////////////////////////////////////////////////////////
template<class K,class T>
void GaussianProcess<K,T>::Predict ( const arma::Mat<T>& inputs,
                                           arma::Row<T>& labels ) const
{
  arma::Mat<T> k_xpx = cov_.GetMatrix(inputs,train_inp_);
  labels = (k_xpx * parameters_).t();
}
///////////////////////////////////////////////////////////////////////////////
template<class K,class T>
void GaussianProcess<K,T>::PredictVariance ( const arma::Mat<T>& inputs,
                                                   arma::Mat<T>& labels )
{
  arma::Mat<T> k_xpx = cov_.GetMatrix(inputs,train_inp_);
  arma::Mat<T> k_xpxp = cov_.GetMatrix(inputs,inputs);
  arma::Mat<T> v = arma::solve(L_,k_xpx.t());
  labels = arma::conv_to<arma::Mat<T>>::from((k_xpxp - v.t()*v));
}
///////////////////////////////////////////////////////////////////////////////
template<class K, class T>
T GaussianProcess<K,T>::ComputeError ( const arma::Mat<T>& inputs,
                                       const arma::Row<T>& labels ) const
{
  arma::Row<T> temp;
  Predict(inputs, temp);
  const size_t n_points = inputs.n_cols;

  temp = labels - temp;

  const T cost = arma::dot(temp, temp) / n_points;

  return cost;
}
///////////////////////////////////////////////////////////////////////////////
template<class K, class T>
T GaussianProcess<K,T>::LogLikelihood ( const arma::Mat<T>& inputs,
                                        const arma::Row<T>& labels ) const
{
  return -0.5*arma::dot(labels,parameters_) - arma::trace(arma::log(L_))
                                            - 0.5*T(N_)*std::log(2.*M_PI);
}
///////////////////////////////////////////////////////////////////////////////
template<class K, class T>
void GaussianProcess<K,T>::SamplePosterior ( const size_t M,
                                             const arma::Mat<T>& inputs,
                                             arma::Mat<T>& labels )
{ 
  arma::Row<T> mean;
  arma::Mat<T> cov;
  Predict(inputs, mean);  
  PredictVariance(inputs, cov);  
  cov.diag() += 1e-6; // jitter addition
  labels = arma::mvnrnd(mean.t(), cov, M).t();
}
///////////////////////////////////////////////////////////////////////////////
template<class K,class T>
void GaussianProcess<K,T>::SamplePrior ( const size_t M,
                                         const arma::Mat<T>& inputs,
                                         arma::Mat<T>& labels ) const
{
  size_t N = inputs.n_cols;
  arma::Mat<T> cov = cov_.GetMatrix(inputs,inputs);  
  cov.diag() += 1e-6 + lambda_; // jitter addition
  arma::Col<T> mean(N); 
  labels = arma::mvnrnd(mean, cov, M).t();
}
///////////////////////////////////////////////////////////////////////////////

} // namespace regression
} // namespace algo

#endif
