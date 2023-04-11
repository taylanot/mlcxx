/**
 * @file covmat.h
 * @author Ozgur Taylan Turan
 *
 * Covariance Matrix creation in armadillo with predefined mlpack kernels
 *
 * TODO: Addition of the diag() get_params(), set_params() and get_grads()
 *
 *
 */

#ifndef COVMAT_H
#define COVMAT_H
// standard
// armadillo
//#include <armadillo>
// local

namespace utils {

template<class T>
struct covmat
{
  T kernel_;
  
  covmat<T>();

  template<typename... Ts>
  covmat<T>(Ts&&... args);

  arma::mat GetMatrix(const arma::mat& input1, const arma::mat& input2) const;
};

} // namespace utils
#include "covmat_impl.h"
#endif 
