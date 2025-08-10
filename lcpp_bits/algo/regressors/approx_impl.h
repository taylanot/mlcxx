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
//-----------------------------------------------------------------------------
// Taylor
//-----------------------------------------------------------------------------
template<class T>
template<class FUNC>
void Taylor<T>::Train (  FUNC& f, const arma::Mat<T>& x0 )
{
  param_.resize(order_+1);
  param_[0] = f.Evaluate(x0).eval()(0);
  for (size_t i=1; i <= order_; i++)
    param_(i) = opt::fdiff(f, x0, type_, h_, order_).eval()(0);
  x0_ = x0;
}
///////////////////////////////////////////////////////////////////////////////
template<class T>
void Taylor<T>::Predict ( const arma::Mat<T>& inputs,
                          arma::Row<T>& labels ) const
{
  arma::Mat<T> temp(order_+1,inputs.n_cols);
  arma::Mat<T> ones = arma::ones<arma::Mat<T>>(1,inputs.n_cols);
  temp.row(0) = ones*param_[0];
  for (size_t i=1; i <= order_; i++)
    temp.row(i) = param_[i] * arma::pow(inputs.each_col()-x0_,order_) 
        / double(arma::cumprod(arma::regspace(1,order_)).eval()(0));
  labels = arma::sum(temp,0); 
}
///////////////////////////////////////////////////////////////////////////////
template<class T>
T Taylor<T>::ComputeError ( const arma::Mat<T>& points, 
                            const arma::Row<T>& responses ) const
{
  arma::Row<T> predictions;
  Predict(points,predictions);
  arma::Row<T> temp =  predictions - responses; 
  return arma::dot(temp,temp) / responses.n_cols;
}
///////////////////////////////////////////////////////////////////////////////
} // namespace approx
} // namespace algo
#endif


