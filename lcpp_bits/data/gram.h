/**
 * @file gram.h
 * @author Ozgur Taylan Turan
 *
 * Matrix creation in armadillo with predefined mlpack kernels
 *
 * TODO: 
 * Addition of the diag() get_params(), set_params() and get_grads()
 * Need to make sure I use only row based kernels
 *       
 *
 *
 */

#ifndef GRAM_H
#define GRAM_H

namespace data {
//-----------------------------------------------------------------------------
// Gram
//-----------------------------------------------------------------------------
template<class KERNEL, class T = DTYPE>
struct Gram
{
    /// Default constructor.
    Gram() {}

    /**
     * @brief Construct and initialize kernel with arbitrary arguments.
     * @tparam Ts Argument types for the kernel constructor.
     * @param args Arguments to forward to the kernel constructor.
     */
    template<typename... Ts>
    Gram(Ts&&... args) : kernel_(args...) {}

    // Kernel instance used for computing Gram matrices.
    KERNEL kernel_;

    /**
     * @brief Compute Gram matrix for row-major ordered data.
     * @param input1 First dataset (rows are samples).
     * @param input2 Second dataset (rows are samples).
     * @return Gram matrix of size input1.n_rows × input2.n_rows.
     */
    arma::Mat<T> GetMatrix2(const arma::Mat<T>& input1,
                            const arma::Mat<T>& input2) const
    {
        arma::Mat<T> matrix(input1.n_rows, input2.n_rows);

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < int(input1.n_rows); i++)
            for (int j = 0; j < int(input2.n_rows); j++)
                matrix(i,j) = kernel_.Evaluate(input1.row(i).eval(),
                                               input2.row(j).eval());

        return matrix;
    }

    /**
     * @brief Compute Gram matrix for column-major ordered data.
     * @param input1 First dataset (columns are samples).
     * @param input2 Second dataset (columns are samples).
     * @return Gram matrix of size input1.n_cols × input2.n_cols.
     */
    arma::Mat<T> GetMatrix(const arma::Mat<T>& input1,
                           const arma::Mat<T>& input2) const
    {
        arma::Mat<T> matrix(input1.n_cols, input2.n_cols);
        
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < int(input1.n_cols); i++)
            for (int j = 0; j < int(input2.n_cols); j++)
                matrix(i,j) = kernel_.Evaluate(input1.col(i).eval(),
                                               input2.col(j).eval());

        return matrix;
    }

    /**
     * @brief Compute an approximate Gram matrix using the Nyström method.
     * 
     * Selects k random landmark points, computes their kernel matrix W,
     * and uses it to approximate the full kernel matrix:
     * K_approx = C * pinv(W) * C^T
     * 
     * @param input1 First dataset (columns are samples).
     * @param input2 Second dataset (columns are samples).
     * @param k      Number of landmark points to sample.
     * @return Approximated Gram matrix.
     */
    arma::Mat<T> GetMatrix_approx(const arma::Mat<T>& input1,
                                  const arma::Mat<T>& input2,
                                  size_t k) const
    {
        size_t n_samples = input1.n_rows;
        arma::uvec indices = arma::randi<arma::uvec>(k, arma::distr_param(0, n_samples - 1));
        arma::Mat<T> landmarks = input1.cols(indices);

        arma::Mat<T> W = this->GetMatrix(landmarks, landmarks);
        arma::Mat<T> C = this->GetMatrix(input1, landmarks);
        arma::Mat<T> W_pinv = arma::pinv(W);

        return C * W_pinv * C.t();
    }

    /**
     * @brief Compute Gram matrix of a dataset with itself (column-major).
     * @param input1 Dataset (columns are samples).
     * @return Symmetric Gram matrix of size input1.n_cols × input1.n_cols.
     */
    arma::Mat<T> GetMatrix(const arma::Mat<T>& input1) const
    {
        return GetMatrix(input1, input1); 
    }
};

} // namespace utils

#endif 
