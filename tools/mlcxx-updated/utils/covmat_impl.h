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
// standard
// mlpack 
//#include <mlpack/core/kernels/cauchy_kernel.hpp>
//#include <mlpack/core/kernels/linear_kernel.hpp>
//#include <mlpack/core/kernels/gaussian_kernel.hpp>
//#include <mlpack/core/kernels/spherical_kernel.hpp>
//#include <mlpack/core/kernels/laplacian_kernel.hpp>
//#include <mlpack/core/kernels/triangular_kernel.hpp>
//#include <mlpack/core/kernels/polynomial_kernel.hpp>
//#include <mlpack/core/kernels/epanechnikov_kernel.hpp>
//#include <mlpack/core/kernels/hyperbolic_tangent_kernel.hpp>
// local
#include "covmat.h"

namespace utils {

template<class T>
covmat<T>::covmat() { }

template<class T>
template<typename... Ts>
covmat<T>::covmat(Ts&&... args): kernel_(args...){ }

template<class T>
arma::mat covmat<T>::GetMatrix(const arma::mat& input1, const arma::mat& input2) const
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



///** DECLARATIONS for available mlpack kernels...
// *
// *  NOTE: if you add another kernel type you must declare it here otherwise you 
// *  will get an error for undefined reference!
//*/
//template class covmat<mlpack::kernel::CauchyKernel>;
//template class covmat<mlpack::kernel::LinearKernel>;
//template class covmat<mlpack::kernel::GaussianKernel>;
//template class covmat<mlpack::kernel::SphericalKernel>;
//template class covmat<mlpack::kernel::LaplacianKernel>;
//template class covmat<mlpack::kernel::TriangularKernel>;
//template class covmat<mlpack::kernel::PolynomialKernel>;
//template class covmat<mlpack::kernel::EpanechnikovKernel>;
//template class covmat<mlpack::kernel::HyperbolicTangentKernel>;

} // namespace utils
