/**
 * @file dists.h
 * @author Ozgur Taylan Turan
 *
 * Random distributions some are not readily available in armadillo
 *
 * TODO: I am torn between seperating this to implementation or keeping it like 
 *        this...
 *
 *
 */

#ifndef DISTS_H
#define DISTS_H

namespace utils {

// Abstract base class for distribution samplers
template<class T>
class Sampler {
private:
    T dist_;
    std::mt19937 gen_;

public:
    // Constructor
    template<class... Ts>
    Sampler ( Ts... args ) : dist_(args...), 
                             gen_() { gen_.seed(SEED); }

    void Fill ( arma::mat& matrix ) 
    {
      for (size_t i = 0; i < matrix.n_rows; ++i)
        for (size_t j = 0; j < matrix.n_cols; ++j)
          matrix(i,j) = dist_(gen_);
    }

    void Fill ( arma::vec& vector ) 
    {
      for (size_t i = 0; i < vector.n_elem; ++i)
          vector[i] = dist_(gen_);
    }

    void Fill ( arma::rowvec& vector ) 
    {
      for (size_t i = 0; i < vector.n_elem; ++i)
          vector[i] = dist_(gen_);
    }

    arma::rowvec Sample ( size_t n ) 
    {
      arma::rowvec samples(n);
      for (size_t i = 0; i < n; ++i)
          samples[i] = dist_(gen_);
      return samples;
    }

    arma::mat Sample(size_t rows, size_t cols)
    {
      arma::mat samples(rows, cols);
      for (size_t i = 0; i < rows; ++i) 
          for (size_t j = 0; j < cols; ++j) 
              samples(i, j) = dist_(gen_);
      return samples;
    }
};

namespace dists {

arma::mat pareto ( const size_t& nrow, const size_t& ncol=1,
                   const double& shp=2., const double& scl=1.,
                   const double& loc=0. ) 
{
  return loc + scl / arma::pow(arma::randu<arma::mat>(nrow,ncol), 1.0 / shp);
}

arma::mat lognorm ( const size_t& nrow, const size_t& ncol=1,
                    const double& sig=1,
                    const double& mu=0. ) 
{
  return arma::exp(mu + sig * arma::randu<arma::mat>(nrow,ncol));
}

arma::mat gaussbimodal ( const size_t& nrow, const size_t& ncol=1,
                         const arma::rowvec& means={-5,5},
                         const arma::rowvec& stds={1,1},
                         const double& prob1=0.5,
                         const size_t& dim=1 )
{
  size_t n, rep;
  if (dim == 1)
  {
    n = ncol;
    rep = nrow;
  }
  else
  {
    n = nrow;
    rep = ncol;
  }

  size_t n1 = size_t(prob1 * n * 0.5);
  size_t n2 = n - n1;  

  arma::rowvec samples(n);
  arma::mat res(rep, n);
  for (size_t i=0; i<rep; i++)
  {
    samples(arma::span(0, n1-1)) = arma::randn(1,n1,
                                          arma::distr_param(means[0],stds[0]));
    samples(arma::span(n1,n-1)) = arma::randn(1,n2,
                                          arma::distr_param(means[1],stds[1]));
    res.row(i) = samples;
  }
  
  if (dim == 0)
    inplace_trans(res);
  return res;
}
/* template<class T> */
/* class Dist */ 
/* { */
/* public: */
/*   virtual ~Dist() {} */

/*   virtual T Sample(size_t n) = 0; */

/*   /1* virtual arma::vec Mean( ) = 0; *1/ */
/*   /1* virtual arma::vec Var( ) = 0; *1/ */
/*   /1* virtual arma::vec Median( ) = 0; *1/ */
/*   /1* virtual arma::vec Mode( ) = 0; *1/ */
/* }; */

/* template<class T> */
/* class Pareto: public Dist<T> */
/* { */
/* private: */
/*   double xm_;     // Scale parameter */
/*   double alpha_;  // Shape parameter */
/*   double loc_;    // Location parameter */

/* public: */
/*   // Constructor */
/*   Pareto ( const double& xm, const double& alpha, const double& loc = 0. ) : */
/*                                                         xm_(xm), */
/*                                                         alpha_(alpha), */
/*                                                         loc_(loc) { } */

/*   T Sample ( size_t n ) override */ 
/*   { */
/*     T u = arma::randu(n); */

/*     return loc_ + xm_ / std::pow(u, 1.0 / alpha_); */
/*   } */
/* }; */

/* template<class T> */
/* class LogNorm: public Dist<T> */
/* { */
/* private: */
/*   double alpha_;  // Shape parameter */
/*   double loc_;    // Location parameter */

/* public: */
/*   // Constructor */
/*   Pareto ( double alpha, double loc = 0. ) : loc_(xm), alpha(alpha) { } */

/*   T Sample ( size_t n ) override */ 
/*   { */
/*     T u = arma::randn(n); */

/*     return loc_ + arma::exp(u); */
/*   } */
/* }; */


} // namespace dists
} // namespace utils
#endif
