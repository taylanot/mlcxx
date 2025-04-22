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

struct RandomSelect
{
  template<class DATASET,class T>
  std::pair<DATASET,DATASET> operator() ( const DATASET& dataset, const T size )
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
struct Bootstrap
{
  template<class DATASET,class T>
  std::pair<DATASET,DATASET> operator()(const DATASET& dataset, const T size)
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

/**
 * Given a dataset, select N of them randomly and keep on taking form the rest
 *
 * @param dataset : dataset to be splited 
 * @param size    : size of the selection
 * @param repeat  : how many times repeat the process
 */
struct Additive
{
  
  template<class DATASET,class T>
  void operator()
  ( const DATASET& dataset, const arma::Row<T> sizes, const size_t repeat,
    std::unordered_map<size_t,std::pair<DATASET,DATASET>>& collect,
    size_t& counter )
  {
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

} // namespace data

#endif
