/**
 * @file quant.cpp
 * @author Ozgur Taylan Turan
 *
 * Here I will try to develop some quantile related stuff
 * (1) qtest -> Alan D. Hutson [A distribution function estimator for the 
 *                difference of order statistics from two independent samples]
 *
 * (2) quant -> Alan D. Hutson [The generalized sigmoidal quantile function]
 *
 */
#include <headers.h>
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
         T qx = 0.5, T qy = 0.5 )
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
    {return boost::math::binomial_coefficient<double>(a,b);};
  auto beta = [](T x, T p, T q) 
    {return boost::math::beta<DTYPE>(p,q,x);};
  auto wj = [&C,&beta](T k,T j,T n) 
    {return j*C(n,j)*(beta(k/n,j,n-j+1.)-beta((k-1.)/n,j,n-j+1.));};

  T sum = 0;
  for (size_t k=1; k<=n; k++)
  {
    sum += beta(F(delta+y[k-1]),i,m-i+1.)*wj(k,j,n);
  }
  return i*C(m,i)*sum;
}

///////////////////////////////////////////////////////////////////////////////
//    Sigmoidal Quantile estimation for small population sizes it is claimed to 
//    perform better than other widely used methods.
//
//    References
//    ----------
//    [1] Alan D. Hutson (2022) The generalized sigmoidal quantile function
//    
///////////////////////////////////////////////////////////////////////////////
template<class VectorType=arma::Row<DTYPE>, class T=DTYPE>
T sigquant ( const VectorType& x )
{
  return 0.;
}

///////////////////////////////////////////////////////////////////////////////
/// MAIN
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
  using Row = arma::Row<DTYPE>;
  using Col = arma::Row<DTYPE>;
  /* arma::Row<DTYPE> x = arma::randu<arma::Row<DTYPE>>(100); */
  /* arma::Row<DTYPE> y = arma::randu<arma::Row<DTYPE>>(100,arma::distr_param(1,2)); */
  /* PRINT_VAR(stats::qtest(0.,x,y)); */
  double sum = 0;
  double h0 = 0;
  double h1a = 0;
  double h1b = 0;
  int rep = 1000;
  size_t N = 10;
  for (int i=0;i<rep;i++)
  {
    auto g1 = arma::randn<Row>(N/2,arma::distr_param(12,1));
    auto g2 = arma::randn<Row>(N/2,arma::distr_param(12,1));
    /* auto g2 = arma::randn<Row>(N/2,arma::distr_param(12,1)); */
    Row d1 = arma::join_rows(g1,g2);
    Row d2 = arma::randn<Row>(N,arma::distr_param(12.,0.01));
    Col qs = {0.95};
    Col q1 = arma::quantile(d1,qs);
    Col q2 = arma::quantile(d2,qs);
    DTYPE delta = q1[0]-q2[0];
    /* sum += delta; */
    

    /* auto v1 = arma::conv_to<std::vector<DTYPE>>::from(d1); */
    /* auto v2 = arma::conv_to<std::vector<DTYPE>>::from(arma::randn<Row>(10,arma::distr_param(10,5))); */

    /* auto [t,p] = boost::math::statistics::one_sample_t_test((d1-d2).eval()); */
    /* auto [t,p] = boost::math::statistics::paired_samples_t_test(v1,v2); */

    auto p = stats::qtest(0.,d1,d2,0.95,0.95);
    sum += p;
    if ((1.-p)<(0.05))
      h1a++;
    else if (p<(0.05))
      h1b++;
    else
      h0++;
  }
  PRINT_VAR("small: "<<h1a/rep);
  PRINT_VAR("big: "<<h1b/rep);
  PRINT_VAR("equal: "<<h0/rep);
  PRINT_VAR(sum/rep);
  return 0;
}
