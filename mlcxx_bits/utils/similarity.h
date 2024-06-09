/**
 * @file similarity.h
 * @author Ozgur Taylan Turan
 *
 * This is the file that will be used for defining similarity measures
 */

#ifndef SIMILARITY_H 
#define SIMILARITY_H


namespace utils {

//=============================================================================
// cos_sim  : get the cosine similarity between two vectors a.b / |a|.|b|
//=============================================================================

/* template<class T> */
/* double cos_sim( const T& vec1, const T& vec2 ) */
/* { */
/*   return arma::norm_dot(vec1,vec2); */
/* } */

arma::rowvec cos_sim ( const arma::mat& mat,
                       const arma::rowvec& vec)
{
  arma::rowvec sims(mat.n_rows);
  for (size_t i=0; i<mat.n_rows; i++)
  {
    arma::rowvec temp = mat.row(i);
    sims(i) = arma::norm_dot(temp,vec);
  }
  return sims;
}

//=============================================================================
// curve_sim  : get the difference between two curves via area calculation
//              curves need to have the same x coordinates. This is only usable
//              for curves that have positive areas (curves defined only for
//              Z^+ -> z^+. Note that, you have to squeeze the similarity scores 
//              between 0 and 1. At this point we assume that we have unit 
//              distance between consecutive points on the curve.
//=============================================================================
arma::rowvec curve_sim( const arma::mat& mat,
                        const arma::rowvec& vec )
{
  return arma::exp(-arma::abs(arma::trapz(mat.each_row()-vec,1).t()));
}

/* arma::rowvec curve_sim2( const arma::mat& mat, */
/*                         const arma::rowvec& vec ) */
/* { */
/*   return arma::sqrt(arma::trapz(arma::pow(mat.each_row()-vec,2),1).t()); */
/* } */
/* arma::rowvec curve_sim( const arma::vec& xt, */ 
/*                         const arma::mat& mat, */
/*                         const arma::rowvec& vec ) */
/* { */
/*   return arma::exp(-arma::abs(arma::trapz(xt,mat.each_row()-vec,1).t())); */
/* } */

} // namespace utils

#endif
