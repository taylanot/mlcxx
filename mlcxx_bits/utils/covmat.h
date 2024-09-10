/**
 * @file covmat.h
 * @author Ozgur Taylan Turan
 *
 * Covariance Matrix creation in armadillo with predefined mlpack kernels
 *
 * TODO: 
 *
 * Addition of the diag() get_params(), set_params() and get_grads()
 * Need to make sure I use only row based kernels
 *       
 *
 *
 */

#ifndef COVMAT_H
#define COVMAT_H

namespace utils {

template<class KERNEL, class T=DTYPE>
struct covmat
{
  covmat() {};

  template<typename... Ts>
  covmat(Ts&&... args): kernel_(args...) { }

  KERNEL kernel_;

  /*
   * This is for row ordered data
   */
  arma::Mat<T> GetMatrix2( const arma::Mat<T>& input1,
                           const arma::Mat<T>& input2 ) const
  {
    arma::Mat<T> matrix(input1.n_rows, input2.n_rows);

    for (int i = 0; i < int(input1.n_rows); i++)
        for (int j = 0; j < int(input2.n_rows); j++)
            matrix(i,j) = kernel_.Evaluate(input1.row(i).eval(),
                                           input2.row(j).eval());

    return matrix;
  }

  /*
   * This is for column ordered data
   */
  arma::Mat<T> GetMatrix ( const arma::Mat<T>& input1,
                           const arma::Mat<T>& input2 ) const
  {
    arma::Mat<T> matrix(input1.n_cols, input2.n_cols);

    for (int i = 0; i < int(input1.n_cols); i++)
        for (int j = 0; j < int(input2.n_cols); j++)
            matrix(i,j) = kernel_.Evaluate(input1.col(i).eval(),
                                           input2.col(j).eval());

    return matrix;
  }
  arma::Mat<T> GetMatrix ( const arma::Mat<T>& input1 ) const
  {
    return GetMatrix(input1, input1); 
  }
};

} // namespace utils

#endif 
