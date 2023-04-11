/**
 * @file functionalpca.h
 * @author Ozgur Taylan Turan
 *
 * Functional PCA
 *
 *
 *
 */
#ifndef FUNCTIONALPCA_H
#define FUNCTIONALPCA_H

namespace algo {
namespace functional {

//-----------------------------------------------------------------------------
// ufpca 
// * Univariate functional pca implementation
// @param inputs    : inputs (DxN)
// @param labels    : labels (MxN) 
//-----------------------------------------------------------------------------
std::tuple<arma::vec, arma::mat> ufpca ( const arma::mat& inputs,
                                         const arma::mat& labels )
{

  arma::vec eigenvalues;
  arma::mat eigenvectors, eigenfunctions;

  size_t D = inputs.n_rows;
  size_t N = inputs.n_cols;
  size_t M = labels.n_rows;

  BOOST_ASSERT_MSG( D == 1, "FunctionalData input dim. != 1!");


  arma::mat weights(1, N);

  weights(0,0)    = inputs(0,1)-inputs(0,0);
  weights(0,N-1)  = inputs(0,inputs.n_cols-1)-inputs(0,inputs.n_cols-2);
  weights(0,arma::span(1,N-2)) = 
                    inputs(0,arma::span(2,N-1)) - inputs(0,arma::span(0,N-3));

  weights *= 0.5;

  arma::mat sqrt_W = arma::diagmat(arma::sqrt(weights));
  arma::mat inv_sqrt_W = arma::diagmat(arma::sqrt(1./weights));
  arma::mat mean = arma::mean(labels,0);
  arma::mat adj_labels(M, N); 

  for(size_t i=0; i < M; i++)
    adj_labels.row(i) = labels.row(i) - mean;

  arma::mat covariance = (adj_labels.t() * adj_labels)/(N-1);
  covariance += covariance.t();
  covariance *= 0.5;
  arma::mat variance = sqrt_W * covariance * sqrt_W;
  
  arma::eig_sym(eigenvalues, eigenvectors, variance);
  eigenvalues = eigenvalues.clamp(0., arma::datum::inf);
  eigenvalues = arma::reverse(eigenvalues);
  eigenvectors = arma::reverse(eigenvectors,1);

  eigenfunctions = (inv_sqrt_W*eigenvectors).t();
  return std::make_tuple(eigenvalues, eigenfunctions);
}


//-----------------------------------------------------------------------------
// ufpca 
// * Univariate functional pca implementation
// @param inputs    : inputs (DxN)
// @param labels    : labels (MxN) 
// @param ppc       : percentage of principle component 
//-----------------------------------------------------------------------------
std::tuple<arma::vec, arma::mat> ufpca ( const arma::mat& inputs,
                                         const arma::mat& labels,
                                         const double ppc )
{
  arma::vec eigenvalues, eigvals;
  arma::mat eigenfunctions, eigfuncs;
  int npc;

  size_t N = inputs.n_cols;

  auto res = ufpca(inputs, labels);
  
  eigenvalues = std::get<0>(res); 
  eigenfunctions = std::get<1>(res); 
  arma::vec cum_ppc= arma::cumsum(eigenvalues) / arma::sum(eigenvalues);

  arma::uvec cum_ppc_index = arma::find( cum_ppc > ppc, 1);
  npc = cum_ppc_index[0];
  
  eigvals = eigenvalues.subvec(arma::span(0,npc-1));

  eigfuncs = eigenfunctions(arma::span(0,npc-1),arma::span(0,N-1));
  return std::make_tuple(eigvals, eigfuncs);
}

//-----------------------------------------------------------------------------
// ufpca 
// * Functional PCA mmplementation for 1D problems
// @param inputs    : inputs (DxN)
// @param labels    : labels (MxN) 
// @param npc       : number of principle component 
//-----------------------------------------------------------------------------
std::tuple<arma::vec, arma::mat> ufpca ( const arma::mat& inputs,
                                         const arma::mat& labels,
                                         const size_t npc )
{

  arma::vec eigenvalues, eigvals;
  arma::mat eigenfunctions, eigfuncs;

  size_t N = inputs.n_cols;

  auto res = ufpca(inputs, labels);
  
  eigenvalues = std::get<0>(res); 
  eigenfunctions = std::get<1>(res); 

  eigvals = eigenvalues.subvec(arma::span(0,npc-1));


  eigfuncs = eigenfunctions(arma::span(0,npc-1),arma::span(0,N-1));
  return std::make_tuple(eigvals, eigfuncs);
}

} // namespace functional
} // namespace utils


#endif
