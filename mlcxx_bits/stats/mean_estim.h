/**
 * @file mean_estim.h
 * @author Ozgur Taylan Turan
 *
 * Mean estimation (mostly for robust (in some cases))
 *
 * TODO: There is unneccessary assignment for now in most of the implementations
 *        you can fix those.
 *
 *
 */

#ifndef MEAN_ESTIM_H
#define MEAN_ESTIM_H

namespace stats {
//=============================================================================
// Emprical: Emprical Mean Objective
//=============================================================================
template<class T=DTYPE>
class Empirical
{
  public:

  Empirical ( ) { } 

  Empirical ( const arma::Mat<T>& X ) : X_(X) { }
  
  arma::Col<T> Evaluate (const arma::Col<T>& mu)
  { 
    return arma::sum(X_.each_col()-mu,1);
  }

  void Gradient ( const arma::Col<T>& mu,
                  arma::Mat<T>& gradient )
  {
    gradient = -double(X_.n_cols);
  }

  private:
  
  arma::Mat<T> X_;

};

//=============================================================================
// Catoni : Catoni's M-estimator Objective 
//=============================================================================
template<class T=DTYPE>
class Catoni
{
  public:

  Catoni ( ) { } 

  Catoni ( const arma::Mat<T>& X,  const double& alpha ) : X_(X), 
                                                        alpha_(alpha) { }
  
  arma::Col<T> Evaluate (const arma::Col<T>& mu)
  { 
    arma::Col<T> v = arma::ones<arma::Col<T>>(1);
    return v * arma::accu(Psi(alpha_*(X_.each_col()-mu)));
  }

  void Gradient ( const arma::Col<T>& mu,
                  arma::Mat<T>& gradient )
  {
    gradient = opt::fdiff(*this, mu);
  }

  arma::Mat<T> Psi( const arma::Mat<T>& x )
  {
    arma::Mat<T> res (arma::size(x));
    arma::uvec _idx = arma::find(x < 0);
    arma::uvec idx_ = arma::find(x >= 0);
    
    // Compute for elements where x < 0
    res(_idx) = -arma::log(1 - x(_idx) + arma::square(x(_idx)) / 2);
    
    // Compute for elements where x >= 0
    res(idx_) = arma::log(1 + x(idx_) + arma::square(x(idx_)) / 2);
    
    return res;
  }

  private:
  
  arma::Mat<T> X_;
  double alpha_;

};

//=============================================================================
// Lee : Lee Objective Function for finding the \alpha
//=============================================================================
template<class T=DTYPE>
class Lee
{
  public:

  Lee ( ) { } 

  Lee ( const arma::Mat<T>& X, const arma::Col<T>& kappa, const double& delta ) : 
                                                          X_(X), 
                                                          kappa_(kappa),
                                                          delta_(delta) 
  { 
    X_kappa_ = X_.each_col()-kappa_;
    const_ = 1./3. * std::log(1/delta_);
  }
  
  arma::Col<T> Evaluate (const arma::Col<T>& alpha)
  { 
    arma::Col<T> vector = arma::ones<arma::Col<T>>(1);
    arma::Row<T> comp(X_.size());

    for ( size_t i=0; i<X_.n_cols;i++ )
    {
      double check = alpha(0)*X_kappa_(i)*X_kappa_(i);
      if ( check < 1.)
        comp[i] = check;
      else
        comp[i] = 1;
    }
    return  vector * (arma::accu(comp) - const_);
  }

  void Gradient ( const arma::Col<T>& alpha,
                  arma::Mat<T>& gradient )
  {
    gradient = opt::fdiff(*this, alpha);
  }

  private:
  
  arma::Mat<T> X_;
  arma::Mat<T> X_kappa_;
  arma::Col<T> kappa_;
  double delta_;
  double const_;
};

//=============================================================================
// emean: Emprical mean estimator
//=============================================================================
template<class T=DTYPE>
arma::Mat<T> emean(const arma::Mat<T>& x, const size_t& dim=1)
{
  arma::Mat<T> x_;
  if (dim == 0)
    x_ = x.t();
  else
    x_ = x;
  Empirical obj(x);
  arma::Col<T> mu = arma::median(x,dim);
  opt::fsolve(obj, mu);
  return mu;
}

//=============================================================================
// catoni: Catoni mean estimator given a variance
//=============================================================================
template<class T=DTYPE>
arma::Mat<T> catoni( const arma::Mat<T>& x,
                     const double& sigma, const size_t& dim=1 )
{
  BOOST_ASSERT_MSG( x.n_cols == 1 || x.n_rows == 1, "Only 1D array please");
  arma::Mat<T> x_;
  if (dim == 0)
    x_ = x.t();
  else
    x_ = x;
  double alpha = 5.*std::sqrt(2./(x_.n_cols*sigma));
  Catoni obj(x,alpha);
  arma::Col<T> mu = arma::median(x,dim);
  opt::fsolve(obj, mu);
  return mu;
}

//=============================================================================
// catoni: Catoni mean estimator with variance estimated from the data
//=============================================================================
template<class T=DTYPE>
arma::Mat<T> catoni(const arma::Mat<T>& x, const size_t& dim=1)
{
  BOOST_ASSERT_MSG( x.n_cols == 1 || x.n_rows == 1, "Only 1D array please");
  // the varinace will be estimated from the variables,
  // chicken egg problem, but what can you do sometimes...
  double sigma = arma::stddev(x,1,dim).eval()(0);
  return catoni(x, sigma, dim);
}

//=============================================================================
// tmean: Trimmed mean estimator
//=============================================================================
template<class T=DTYPE>
arma::Mat<T>  tmean(const arma::Mat<T>& x, const size_t& n=1, const size_t& dim=1) 
{
  arma::Mat<T> x_;
  if (dim == 0)
      x_ = x.t();
  else
      x_ = x;
   return arma::mean(
    arma::sort(x_,"ascend",dim).eval().cols(arma::span(n,x_.n_cols-n-1)),dim);
}

//=============================================================================
// ksplit: split a matrix into k columned batches
//=============================================================================
template<class T=DTYPE>
arma::field<arma::Mat<T>> k_split ( const arma::Mat<T>& matrix,
                                    const size_t& k ) 
{
  size_t n_cols = matrix.n_cols;
  arma::field<arma::Mat<T>> groups(k);

  // Calculate the size of each group
  std::vector<size_t> sizes(k);
  size_t base_size = n_cols / k;
  size_t remainder = n_cols % k;

  for (size_t i = 0; i < k; ++i)
      sizes[i] = base_size + (i < remainder ? 1 : 0);

  // Split each column of the matrix based on the calculated sizes
  size_t start = 0;
  for (size_t i = 0; i < k; ++i) 
  {
    size_t end = start + sizes[i];
    groups(i) = matrix.cols(start, end - 1);
    start = end;
  }

  return groups;
}

//=============================================================================
// mediomean: Median-of-Means mean estimator
//=============================================================================
template<class T=DTYPE>
arma::Mat<T> mediomean(const arma::Mat<T>& x, const size_t& k=2, const size_t& dim=1) 
{
  arma::Mat<T> x_;
  if (dim == 0)
      x_ = x.t();
  else
      x_ = x;

  auto splits = k_split(x,k);

  arma::Mat<T> res(1,splits.n_elem);
  for (size_t i=0; i<splits.n_elem; i++)
    res(0,i) = arma::mean(splits[i].eval(),1).eval()(0,0);

  return arma::median(res,1);
}

//=============================================================================
// lee: Lee mean estimator (Optimal Sub-Gaussian Mean Estimation in \R)
//=============================================================================
template<class T=DTYPE>
arma::Mat<T> lee (const arma::Mat<T>& x, const double& delta=0.01, const size_t& dim=1) 
{
  BOOST_ASSERT_MSG( x.n_cols == 1 || x.n_rows == 1, "Only 1D array please");

  size_t k = std::log(1./delta);
  arma::Col<T> kappa = mediomean(x,k,dim);
  arma::Mat<T> x_;

  if (dim == 0)
      x_ = x.t();
  else
      x_ = x;

  arma::Col<T> alpha(1);
  Lee obj(x_,kappa,delta);
  opt::fsolve(obj,alpha);
  arma::Mat<T> xkappa = x_.each_col()-kappa;
  arma::Mat<T> comp(1,x_.size());

  for ( size_t i; i<x_.n_cols;i++ )
  {
    double check = alpha(0)*xkappa[i]*xkappa[i];
    if (check < 1)
      comp[i] = 1-check;
    else
      comp[i] = 0;
  }
  return kappa + arma::accu(xkappa % comp) / x.n_cols;
}
} // namespace stats
#endif
