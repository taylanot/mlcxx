/**
 * @file approx_impl.h
 * @author Ozgur Taylan Turan
 *
 * Function Approximations
 *
 */

#ifndef APPROX_IMPL_H
#define APPROX_IMPL_H

namespace algo { 
namespace approx {


template<class FUNC>
void Taylor::Train (  FUNC& f, const arma::mat& x0 )
{
  param_.resize(order_+1);
  param_[0] = f.Evaluate(x0).eval()(0);
  for (size_t i=1; i <= order_; i++)
    param_(i) = utils::fdiff(f, x0, type_, h_, order_).eval()(0);
  x0_ = x0;
}

void Taylor::Predict ( const arma::mat& inputs,
                             arma::rowvec& labels ) const
{
  arma::mat temp(order_+1,inputs.n_cols);
  arma::mat ones = arma::ones(1,inputs.n_cols);
  temp.row(0) = ones*param_[0];
  for (size_t i=1; i <= order_; i++)
    temp.row(i) = param_[i] * arma::pow(inputs.each_col()-x0_,order_) 
        / double(arma::cumprod(arma::regspace(1,order_)).eval()(0));
  labels = arma::sum(temp,0); 
}

double Taylor::ComputeError ( const arma::mat& points, 
                              const arma::rowvec& responses ) const
{
  arma::rowvec predictions;
  Predict(points,predictions);
  arma::rowvec temp =  predictions - responses; 
  return arma::dot(temp,temp) / responses.n_cols;
}
} // namespace approx
} // namespace algo
#endif


