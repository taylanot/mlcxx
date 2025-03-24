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
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < int(input1.n_cols); i++)
        for (int j = 0; j < int(input2.n_cols); j++)
            matrix(i,j) = kernel_.Evaluate(input1.col(i).eval(),
                                           input2.col(j).eval());

    return matrix;
  }

  /*
   * This is for column ordered data
   */
  arma::Mat<T> GetMatrix_approx ( const arma::Mat<T>& input1,
                                  const arma::Mat<T>& input2, size_t k ) const
  {
    size_t n_samples = input1.n_rows;

    arma::uvec indices = arma::randi<arma::uvec>(k,arma::distr_param(0,n_samples-1));

    arma::Mat<T> landmarks = input1.cols(indices);

    // Compute W and C
    arma::Mat<T> W = this->GetMatrix(landmarks, landmarks);
    arma::Mat<T> C = this->GetMatrix(input1, landmarks);

    // Compute pseudoinverse of W
    arma::Mat<T> W_pinv = arma::pinv(W);

    // Approximated kernel matrix: K_approx = C * W_pinv * C^T
    arma::Mat<T> K_approx = C * W_pinv * C.t();

    return K_approx;
  }

  arma::Mat<T> GetMatrix ( const arma::Mat<T>& input1 ) const
  {
    return GetMatrix(input1, input1); 
  }
};

} // namespace utils

#endif 
