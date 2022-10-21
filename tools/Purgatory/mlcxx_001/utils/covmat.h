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
#include <armadillo>
// local

namespace utils {

template<class T>
struct covmat
{
  T kernel;

  arma::mat matrix;
  
  covmat<T>();
  template<typename... Types>
  covmat<T>(const arma::mat& input1, Types&&... args);
  template<typename... Types>
  covmat<T>(const arma::mat& input1, const arma::mat& input2, Types&&... args);

  covmat<T>(const arma::mat& input1);
  covmat<T>(const arma::mat& input1, const arma::mat& input2);

  void create(const arma::mat& input1, const arma::mat& input2);
};

template<class T>
arma::mat covar(T kernel, const arma::mat& input1)
{
  arma::mat matrix(input1.n_rows, input1.n_rows);

  for (int i = 0; i < int(input1.n_rows); i++)
  {
      for (int j = 0; j < int(input1.n_rows); j++)
      {
        matrix(i,j) = kernel.Evaluate(input1.row(i),input1.row(j));
      }
  }

  return matrix;

}
} // namespace utils
#endif 
