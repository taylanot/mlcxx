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
//    Hypothesis Testing for Quantile differences
//
//    H0 : Q(qx)-P(qy) = delta
//    H1A: Q(qx)-P(qy) < delta
//    H1B: Q(qx)-P(qy) > delta
//
//    * 1-pvalue < alpha/2 -> accept H1B
//    * pvalue < alpha/2   -> accept H1A
//    * otherwise          -> accept H0
//    References
//    ----------
//    [1]  Alan D. Hutson (2009). A distribution function estimator for 
//            the difference of order statistics from two independent samples
//    
///////////////////////////////////////////////////////////////////////////////
template<class VectorType=arma::Row<DTYPE>, class T=DTYPE>
T qtest( T delta, const VectorType& x, const VectorType& y,
         const T qx = 0.5, const T qy = 0.5 )
{
  using ECDF = 
    boost::math::empirical_cumulative_distribution_function<std::vector<DTYPE>>;
  ECDF F(arma::conv_to<std::vector<DTYPE>>::from(x));
  // Get the sizes of the populations
  size_t m = x.n_elem;
  size_t n = y.n_elem;
  // Get the order statistic rounded to the nearest integer 
  size_t i = std::ceil(qx*m)-1;
  size_t j = std::ceil(qy*n)-1;
  // Combination calculator 
  auto C = [](size_t a, size_t b) 
    {return boost::math::binomial_coefficient<T>(a,b);};
  auto beta = [](T x, T p, T q) 
    {return boost::math::beta<T>(p,q,x);};
  auto wj = [&C,&beta,&j,&n](T k) 
    {return j*C(n,j)*(beta(k/n,j,n-j+1.)-beta((k-1.)/n,j,n-j+1.));};
  /* auto wj = [&C,&beta](const T k,const T j, const T n) */ 
  /*   {return j*C(n,j)*(beta(k/n,j,n-j+1.)-beta((k-1.)/n,j,n-j+1.));}; */
  T sum = 0;
  for (size_t k=1; k<=n; k++)
  {
    sum += beta(F(delta+y[k-1]),i,m-i+1.)*wj(k);
  }
  return i*C(m,i)*sum;
}


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
