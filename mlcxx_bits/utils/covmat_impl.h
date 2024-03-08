/**
 * @file covmat_impl.h
 * @author Ozgur Taylan Turan
 *
 * Covariance Matrix creation in armadillo with predefined mlpack kernels
 *
 * TODO: Addition of the diag() get_params(), set_params() and get_grads()
 *
 *
 */
#include "covmat.h"

namespace utils {

template<class T>
covmat<T>::covmat() { }

template<class T>
template<typename... Ts>
covmat<T>::covmat(Ts&&... args): kernel_(args...){ }

template<class T>
arma::mat covmat<T>::GetMatrix ( const arma::mat& input1,
                                 const arma::mat& input2 ) const
{
  arma::mat matrix(input1.n_rows, input2.n_rows);

  for (int i = 0; i < int(input1.n_rows); i++)
    {
      for (int j = 0; j < int(input2.n_rows); j++)
        {
          matrix(i,j) = kernel_.Evaluate(input1.row(i),input2.row(j));
        }
    }

  return matrix;
}


} // namespace utils
