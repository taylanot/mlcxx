/**
 * @file sample.h
 * @author Ozgur Taylan Turan
 *
 * Samplers for datasets.
 *
 */

#ifndef SAMPLE_H
#define SAMPLE_H

namespace data {
//-----------------------------------------------------------------------------
// RandomSelect : Given a dataset, select N of them randomly without replacement 
// and seperate the rest.
//-----------------------------------------------------------------------------
/**
 *
 * @param dataset : dataset to be splited 
 * @param size    : size of the selection
 * @param repeat  : how many times repeat the process
 */
template<class O=arma::uword>
struct RandomSelect
{
  template<class T>
  std::pair<arma::Col<O>,arma::Col<O>> operator() 
  ( const arma::Col<O>& total, const size_t size, const T N )
  {
    arma::Col<O> a,b;
    Split(total,a,b,N);
    return std::pair<arma::Col<O>,arma::Col<O>>(a,b);
  }

  void operator() ( const size_t size,
                    const arma::Row<size_t> Ns,
                    const size_t repeat,
                    std::vector<std::pair<arma::Col<O>,arma::Col<O>>>& collect,
                    const size_t seed = SEED )
	{
    arma::Col<O> total = arma::regspace<arma::Col<O>>(0,size-1);
    mlpack::RandomSeed(seed);
    O counter = 0;
    collect.clear();
    collect.resize(repeat*Ns.n_elem);
    for (size_t j = 0; j < repeat; ++j)
      for (size_t i = 0; i < Ns.n_elem; ++i)
        collect.at(counter++) = (*this)(total, size, Ns[i]); 
	}
};
//-----------------------------------------------------------------------------
// Bootstrap : Given a dataset, select N of them randomly with replacement and
// seperate the rest.
//-----------------------------------------------------------------------------
/**
 * @param dataset : dataset to be splited 
 * @param size    : size of the selection
 * @param repeat  : how many times repeat the process
 * @param collect : collection of your sets
 * @param counter : for labeling your sets
 */
template<class O=arma::uword>
struct Bootstrap
{
  template<class T>
  std::pair<arma::Col<O>,arma::Col<O>> operator() 
  ( const arma::Col<O> total, const size_t size, const T N )
  {
    auto sel = arma::randi<arma::uvec>(N, arma::distr_param(0,size-1));
    sel = arma::sort(sel);
    auto rest = data::SetDiff(total,sel);
    return std::pair<arma::Col<O>,arma::Col<O>>(sel,rest);
  }

  void operator() ( const size_t size,
                    const arma::Row<size_t> Ns,
                    const size_t repeat,
                    std::vector<std::pair<arma::Col<O>,arma::Col<O>>>& collect,
                    const size_t seed = SEED )
	{
    arma::Col<O> total = arma::regspace<arma::Col<O>>(0,size-1);
    mlpack::RandomSeed(seed);
    O counter = 0;
    collect.clear();
    collect.resize(repeat*Ns.n_elem);
    for (size_t j = 0; j < repeat; ++j)
      for (size_t i = 0; i < Ns.n_elem; ++i)
        collect.at(counter++) = (*this)(total, size, Ns[i]); 
	}
};
//-----------------------------------------------------------------------------
// Additive : Given a dataset, select N of them randomly and keep on taking
// form the rest.
//-----------------------------------------------------------------------------
/**
 * @param dataset : dataset to be splited 
 * @param size    : size of the selection
 * @param repeat  : how many times repeat the process
 */
template<class O=arma::uword>
struct Additive
{
  void operator() ( const size_t size,
                    const arma::Row<size_t> Ns,
                    const size_t repeat,
                    std::vector<std::pair<arma::Col<O>,arma::Col<O>>>& collect,
                    const size_t seed = SEED )
  {
    size_t counter = 0;
    auto total = arma::regspace<arma::Col<O>>(0,size-1);
    collect.clear();
    collect.resize(repeat*Ns.n_elem);
    mlpack::RandomSeed(seed);
    for (size_t j = 0; j < repeat; ++j)
    {
      arma::Col<O> trainset,testset;
      for (size_t i=0; i < Ns.n_elem; i++)
      {
        if (i == 0)
          Split(total,trainset,testset,Ns[i]);
        else
          Migrate(trainset,testset,Ns[i]-Ns[i-1]);
        collect.at(counter++) = 
            std::pair<arma::Col<O>,arma::Col<O>>(trainset,testset);
      }
    }
  }
};

} // namespace data

#endif
