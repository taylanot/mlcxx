/**
 * @file highmoments.h
 * @author Ozgur Taylan Turan
 *
 */
#ifndef HIGH_MOMENTS_H
#define HIGH_MOMENTS_H

namespace stats {
//=============================================================================
// Skew  : get the skewness of the data in direction
//=============================================================================
template<class T=DTYPE>
arma::Col<T> Skew(const arma::Mat<T>& data, size_t dim=1)
{
  arma::Mat<T> diff;
  if (dim == 1)
    diff = data.each_col() - arma::mean(data,dim);
  else 
    diff = data.each_row() - arma::mean(data,dim);


  return arma::mean(arma::pow(diff,3), dim) 
        / arma::pow(arma::sqrt(arma::var(data,1,dim)), 3);
}
//=============================================================================
// Kurt  : get the kurtosis of the data in direction
//=============================================================================
template<class T=DTYPE>
arma::Col<T> Kurt(const arma::Mat<T>& data, bool fisher=true,size_t dim=1)
{
  arma::Mat<T> diff;
  if (dim == 1)
    diff = data.each_col() - arma::mean(data,dim);
  else 
    diff = data.each_row() - arma::mean(data,dim);

  arma::Col<T> res = arma::mean(arma::pow(diff,4), dim) 
        / arma::pow(arma::var(data,1,dim),2);
  if (fisher)
    return res-3.0;
  else
    return res;
}


} // namespace stats

#endif
