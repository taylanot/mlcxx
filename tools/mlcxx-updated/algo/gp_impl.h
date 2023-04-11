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

//=============================================================================
// Gaussian Process Regression
//=============================================================================
template<class T>
template<class... Ts>
GaussianProcess<T>::GaussianProcess ( const arma::mat& inputs,
                                      const arma::rowvec& labels,
                                      const double& lambda,
                                      const Ts&... args ) :
                                   cov_(args...), lambda_(lambda)
{
  Train(inputs, labels);
}


template<class T>
void GaussianProcess<T>::Train ( const arma::mat& inputs,
                                 const arma::rowvec& labels )
{
  train_inp_ = inputs.t();
  N_ = inputs.n_cols;
  
  arma::mat k_xx = cov_.GetMatrix(train_inp_,train_inp_);

  arma::mat KLambda = k_xx+
       (lambda_ + 1.e-6) * arma::eye<arma::mat>(k_xx.n_rows, k_xx.n_rows);

  L_ = arma::chol(KLambda, "lower");
  parameters_ = arma::solve(L_.t(), arma::solve(L_,labels.t()));
}

template<class T>
void GaussianProcess<T>::Predict ( const arma::mat& inputs,
                                   arma::rowvec& labels ) const
{
  arma::mat k_xpx = cov_.GetMatrix(inputs.t(),train_inp_);
  labels = (k_xpx * parameters_).t();
}

template<class T>
void GaussianProcess<T>::PredictVariance ( const arma::mat& inputs,
                                           arma::rowvec& labels ) const
{
  arma::mat k_xpx = cov_.GetMatrix(inputs.t(),train_inp_);
  arma::mat k_xpxp = cov_.GetMatrix(inputs.t(),inputs.t());
  arma::vec v = arma::solve(L_,k_xpx.t());
  labels = arma::conv_to<arma::rowvec>::from
                                          ((k_xpxp - v.t()*v));
}

template<class T>
void GaussianProcess<T>::PredictVariance ( const arma::mat& inputs,
                                           arma::mat& labels ) const
{
  arma::mat k_xpx = cov_.GetMatrix(inputs.t(),train_inp_);
  arma::mat k_xpxp = cov_.GetMatrix(inputs.t(),inputs.t());
  arma::mat v = arma::solve(L_,k_xpx.t());
  labels = k_xpxp - v.t()*v;
}

template<class T>
double GaussianProcess<T>::ComputeError ( const arma::mat& inputs,
                                          const arma::rowvec& labels ) const
{
  arma::rowvec temp;
  Predict(inputs, temp);
  const size_t n_points = inputs.n_cols;

  temp = labels - temp;

  const double cost = arma::dot(temp, temp) / n_points;

  return cost;
}

template<class T>
double GaussianProcess<T>::LogLikelihood ( const arma::mat& inputs,
                                           const arma::rowvec& labels ) const
{
  return -0.5*arma::dot(labels,parameters_) - arma::trace(arma::log(L_))
                                            - 0.5*double(N_)*std::log(2.*M_PI);
}

template<class T>
void GaussianProcess<T>::SamplePosterior ( const size_t& M,
                                           const arma::mat& inputs,
                                           arma::mat& labels ) const
{ 
  arma::rowvec mean;
  arma::mat cov;
  Predict(inputs, mean);  
  PredictVariance(inputs, cov);  
  cov.diag() += 1e-6; // jitter addition
  labels = arma::mvnrnd(mean.t(), cov, M).t();
}

template<class T>
void GaussianProcess<T>::SamplePrior ( const size_t& M,
                                       const arma::mat& inputs,
                                       arma::mat& labels ) const
{
  size_t N = inputs.n_cols;
  arma::mat cov = cov_.GetMatrix(inputs.t(),inputs.t());  
  cov.diag() += 1e-6 + lambda_; // jitter addition
  arma::vec mean(N); 
  labels = arma::mvnrnd(mean, cov, M).t();
}

} // namespace regression
} // namespace algo

#endif
