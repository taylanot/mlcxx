/**
 * @file hypothesis.h
 * @author Ozgur Taylan Turan
 * 
 * Hypothesis Tests 
 */
#ifndef HYPOTHESIS_H 
#define HYPOTHESIS_H

namespace stats {
///////////////////////////////////////////////////////////////////////////////
//    rankdata
//    * Ranks the data in a vector 
//    * TODO: 
//      - Add a NaN filter.
//      - Add other methods.
//      - Add other types.
///////////////////////////////////////////////////////////////////////////////
arma::rowvec _rankdata( const arma::rowvec& x,
                        const std::string& method = "average" )
{
  arma::rowvec x_sorted, ranked;

  size_t nx = x.n_cols;
  
  bool sorted = x.is_sorted();

  if ( !sorted )
    x_sorted = arma::sort(x);
  else
    x_sorted = x;

  ranked.resize(nx); 

  if ( method == "average" )
  {
    size_t tot = 0; // Stack the ranks
    size_t freq = 0; // Frequency of the data
    size_t rank = 1;
    double prev = x_sorted(0); // Stack the ranks

    std::map<double, double> maprank;

    for ( size_t i=0; i < nx; i++ )
    {
      if ( prev == x_sorted(i) )
      {
        freq++; tot+=rank;
      }
      else
      {
        double now = 0.;
        now = (double) tot/freq;
        maprank[prev] = now;
        prev = x_sorted(i);
        tot = rank;
        freq = 1;
      }
      rank++;
    }
    maprank[x_sorted(nx-1)] = (double) tot / freq;
    for ( size_t i=0; i < nx; i++ )
      ranked(i) = maprank[x(i)]; 
  }
  return ranked;
}

///////////////////////////////////////////////////////////////////////////////
//    _terms
//    * Used in _cdf_cvm_inf for obtaining an expression that needs the std
//    library math functions 
///////////////////////////////////////////////////////////////////////////////
arma::rowvec _terms( const arma::rowvec& x, size_t k )
{
  size_t nx = x.n_cols;
  arma::rowvec res(nx);
  double u, y, q, b;

  y = 4*k + 1;

  for ( size_t i=0; i < nx; i++ )
  {
    u = exp( std::lgamma(k+0.5) - lgamma(k+1) ) / 
                                            ( pow(M_PI,1.5) * sqrt(x(i)) );
    q = pow(y,2) / ( 16*x(i) );
    b = std::cyl_bessel_k (0.25,q);
    res(i) = u * sqrt(y) * exp(-q) * b;
  }
  return res;  
}
/////////////////////////////////////////////////////////////////////////////////
////    _cdf_cvm_inf
////    * Calculate the exact p-value of the Cramer-von Mises two-sample test.
/////////////////////////////////////////////////////////////////////////////////
//double _exact_cvm_2samp_pval ( const double& s, const double& m, 
//                               const double& n )
//{
//  double lcm, a , b, zeta, zeta_bnd, comb, max_gs;
//
//  lcm = std::lcm ((int) m, (int) n);
//
//  a = floor(lcm/m); b = floor(lcm/n);
//  
//  double exp1 = pow(lcm, 2) * (m + n) * (6 * s - (m*n) * ((4 * m *n) - 1));
//  double exp2 = pow(6*m*n,2);
//  zeta = floor( exp1 / exp2 );
//
//  zeta_bnd = pow(lcm,2) * (m+n);
//
//  comb = boost::math::factorial<double> (m+n) /
//  ( boost::math::factorial<double> (m) *  boost::math::factorial<double> (n) );
//
//  max_gs = std::max( zeta_bnd, comb );
//
//    
//  std::map<int, arma::imat> gs;
//
//  for( int i=0; i < m; i++ )
//  { 
//    arma::imat tmp;
//    gs[i] = tmp;
//  }
//  arma::imat first = "0;1";
//  gs[0] = first;
//
//  int g;
//  arma::umat ind;
//  int val;
//
//  for( int u=0; u < n+1; u++ )
//  {
//    arma::imat temp(2,1);
//    
//    for ( int v=0; v < m; v++ )
//    {
//      g = gs[v];
//      if ( temp(0) == g(0) )
//      {
//        tmp = k
//
//
//      //g = gs[v];
//      //ind = arma::find(temp(0),g);
//
//
//      //if (vi(0) == 0)
//      //  temp = arma::join_rows(inter,tmp(k
//    }
//  }
//
//
//  return 0;
//}
///////////////////////////////////////////////////////////////////////////////
//    _cdf_cvm_inf
//    * Calculate the cdf of the Cramér-von Mises statistic. (Inf. sample size)
//    See equation 1.2 in Csörgő, S. and Faraway, J. (1996).
//    The function is not expected to be accurate for large values of x, say
//    x > 4, when the cdf is very close to 1.
///////////////////////////////////////////////////////////////////////////////
arma::rowvec _cdf_cvm_inf ( const arma::rowvec& x )
{
  size_t nx = x.n_cols;
  arma::rowvec tot(nx); tot.zeros();
  arma::rowvec z;
  bool status = true;
  int k = 0;
  arma::uvec index = arma::regspace<arma::uvec>(0,1,nx-1);
  while ( status )
  {
    z = _terms(x.elem(index).t(), k);
    tot(index) += z;
    index = arma::find( arma::abs(z) > 1e-7 );
    status = arma::any( arma::abs(z) >= 1e-7 );
    k++;
  } 
  return tot;  
}
///////////////////////////////////////////////////////////////////////////////
//    cramervonmisses_2samp
//    * 2 Sample Cramer von Missess Test returning a p value.
//    * TODO:
//      - Add Exact calculation for smaller populations!
//
//    References
//    ----------
//    [1] https://en.wikipedia.org/wiki/Cramer-von_Mises_criterion
//    [2] Anderson, T.W. (1962). On the distribution of the two-sample
//        Cramer-von-Mises criterion. The Annals of Mathematical
//        Statistics, pp. 1148-1159.
//    [3] Conover, W.J., Practical Nonparametric Statistics, 1971.
//    
///////////////////////////////////////////////////////////////////////////////
double cramervonmisses_2samp( const arma::rowvec& x, const arma::rowvec& y, 
                              const std::string& method = "asymptotic" )
{
  double p_val; size_t nx, ny;
  std::string method_upd;

  nx = x.n_cols;ny = y.n_cols;

  arma::rowvec x_sorted = arma::sort(x);
  arma::rowvec y_sorted = arma::sort(y);  
  
  if ( method == "auto" )
  {
    if ( std::max(nx,ny)  > 20 )
      method_upd = "asymptotic";
    else
      method_upd = "exact";
  }
  else
    method_upd = method;

  BOOST_ASSERT_MSG( method == "asymptotic", "EXACT IS NOT IMPLEMENTED YET!" );

  arma::rowvec r, rx, ry;

  r = _rankdata(arma::join_rows(x,y));
  rx = r.cols(0,nx-1); ry = r.cols(nx,nx+ny-1);
 
  double u = nx * arma::sum( arma::pow((rx -
                                 arma::regspace<arma::rowvec>(1, 1, nx)),2) );
  u += ny * arma::sum( arma::pow((ry -
                                 arma::regspace<arma::rowvec>(1, 1, ny)),2) );
  
  double k, N, t;

  k = nx*ny; N = nx + ny;
  t = u /  (k*N) - (4*k - 1)/(6*N);

  size_t et = (1 + 1/N)/6;
  double vt = (N+1) * (4*k*N - 3*(pow(nx,2) + pow(ny,2)) - 2*k);
  vt = vt / (45 * pow(N,2) * 4 * k);

  double tn = 1/6 + (t - et) / sqrt(45 * vt);
  if ( tn < 0.003 )
    p_val = 1.0;
  else
  {
    arma::rowvec term = _cdf_cvm_inf((arma::rowvec) tn);
    p_val = std::max(0., 1.-term(0));
  }

  return p_val;
}

} // namespace stats

#endif
