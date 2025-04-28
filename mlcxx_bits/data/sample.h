/**
 * @file sample.h
 * @author Ozgur Taylan Turan
 *
 * Samplers for datasets
 *
 */

#ifndef SAMPLE_H
#define SAMPLE_H

namespace data {
/**
 * Given a dataset, select N of them randomly without replacement and seperate 
 * the rest
 *
 * @param dataset : dataset to be splited 
 * @param size    : size of the selection
 * @param repeat  : how many times repeat the process
 */

struct RandomSelect_
{
  template<class DATASET,class T>
  std::pair<DATASET,DATASET> operator() ( const DATASET& dataset, const T size,
                                          const size_t seed = SEED )
  {
    DATASET trainset,testset;
    Split(dataset,trainset,testset,size);
    return std::pair<DATASET&,DATASET&>(trainset,testset);
  }

  template<class DATASET,class T>
  void operator()
  ( const DATASET& dataset, const arma::Row<T> sizes, const size_t repeat,
    std::unordered_map<size_t,std::pair<DATASET,DATASET>>& collect,
    size_t& counter )
	{
    for (size_t j = 0; j < repeat; ++j)
      for (size_t i = 0; i < sizes.n_elem; ++i)
        collect[counter++] = (*this)(dataset, sizes[i]); 
	}

};

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

/**
 * Given a dataset, select N of them randomly with replacement and seperate the 
 * rest.
 *
 * @param dataset : dataset to be splited 
 * @param size    : size of the selection
 * @param repeat  : how many times repeat the process
 * @param collect : collection of your sets
 * @param counter : for labeling your sets
 */
struct Bootstrap_
{
  template<class DATASET,class T>
  std::pair<DATASET,DATASET> operator()( const DATASET& dataset, const T size )
  {
    DATASET trainset,testset;
    auto all = arma::regspace<arma::uvec>(0,1,dataset.labels_.n_cols-1);
    auto sel = arma::randi<arma::uvec>(size,
                                arma::distr_param(0,dataset.labels_.n_cols-1));
    sel = arma::sort(sel);
    trainset.inputs_ = dataset.inputs_.cols(sel); 
    trainset.labels_ = dataset.labels_.cols(sel); 
    auto rest = data::SetDiff(all,sel);
    testset.inputs_ = dataset.inputs_.cols(rest); 
    testset.labels_ = dataset.labels_.cols(rest); 

    return std::pair<DATASET,DATASET>(trainset,testset);
  }

  template<class DATASET,class T>
  void operator()
  ( const DATASET& dataset, const arma::Row<T> sizes, const size_t repeat,
    std::unordered_map<size_t,std::pair<DATASET,DATASET>>& collect,
    size_t& counter )
	{
    for (size_t j = 0; j < repeat; ++j)
      for (size_t i = 0; i < sizes.n_elem; ++i)
        collect[counter++] = (*this)(dataset, sizes[i]); 
	}
};

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

/**
 * Given a dataset, select N of them randomly and keep on taking form the rest
 *
 * @param dataset : dataset to be splited 
 * @param size    : size of the selection
 * @param repeat  : how many times repeat the process
 */
struct Additive_
{
  
  template<class DATASET,class T>
  void operator()
  ( const DATASET& dataset, const arma::Row<T> sizes, const size_t repeat,
    std::unordered_map<size_t,std::pair<DATASET,DATASET>>& collect,
    size_t& counter, const size_t seed = SEED  )
  {
    mlpack::RandomSeed(seed);
    for (size_t j = 0; j < repeat; ++j)
    {
      DATASET trainset,testset;
      for (size_t i=0; i < sizes.n_elem; i++)
      {
        if (i == 0)
          Split(dataset,trainset,testset,sizes[i]);
        else
          Migrate(dataset,trainset,testset,sizes[i]-sizes[i-1]);
        collect[counter++] = std::pair<DATASET,DATASET>(trainset,testset);
      }
    }
  }
};

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
          Migrate(total,trainset,testset,Ns[i]-Ns[i-1]);
        collect.at(counter++) = 
            std::pair<arma::Col<O>,arma::Col<O>>(trainset,testset);
      }
    }
  }
};

} // namespace data

#endif
