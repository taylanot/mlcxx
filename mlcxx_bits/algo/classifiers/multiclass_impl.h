/**
 * @file multiclass_impl.h
 * @author Ozgur Taylan Turan
 *
 * MultiClass Classifier for binary classifiers
 *
 */

#ifndef MULTICLASS_IMPL_H
#define MULTICLASS_IMPL_H

namespace algo {
namespace classification {

template<class MODEL, class T>
template<class... Args>
OnevAll<MODEL,T>::OnevAll( const size_t& num_class, const Args&... args ) 
{ 
  models_.resize(num_class);
  for(size_t i=0;i<num_class;i++)
    models_[i] = MODEL(args...);
}

template<class MODEL, class T>
template<class... Args>
OnevAll<MODEL,T>::OnevAll( const arma::Mat<T>& inputs,
                           const arma::Row<size_t>& labels,
                           const size_t& num_class,
                           const Args&... args ) : nclass_(num_class)
{ 
  unq_ = arma::unique(labels).eval();
  if (unq_.n_elem == 1)
    oneclass_ = true;

  unclass_ = unq_.n_elem;
  models_.resize(unclass_);

  for(size_t i=0;i<unclass_;i++)
    models_[i] = MODEL(args...);

  Train(inputs, labels, args...);
  
} 

template<class MODEL, class T>
template<class... Args>
void OnevAll<MODEL,T>::Train ( const arma::Mat<T>& inputs,
                               const arma::Row<size_t>& labels ) 
{ 
  if (!oneclass_)
  {
    for(size_t i=0;i<unclass_;i++)
    {
      auto binlabels = arma::conv_to<arma::Row<size_t>>::from(labels==unq_(i));
      models_[i].Train(inputs, binlabels);
    }
  }
  else
    return;
}

template<class MODEL, class T>
template<class... Args>
void OnevAll<MODEL,T>::Train ( const arma::Mat<T>& inputs,
                               const arma::Row<size_t>& labels,
                               const Args&... args ) 
{ 
  if (!oneclass_)
  {
    models_.resize(unq_.n_elem);
    for(size_t i=0;i<unclass_;i++)
    {
      auto binlabels = arma::conv_to<arma::Row<size_t>>::from(labels==unq_(i));
      MODEL model(inputs,binlabels,args...);
      models_[i] = model;
    }
  }
} 

template<class MODEL, class T>
void OnevAll<MODEL,T>::Classify( const arma::Mat<T>& inputs,
                                 arma::Row<size_t>& preds )
{
  arma::Mat<T> probs;
  Classify(inputs, preds, probs);
}

template<class MODEL, class T>
void OnevAll<MODEL,T>::Classify( const arma::Mat<T>& inputs,
                                 arma::Row<size_t>& preds,
                                 arma::Mat<T>& probs )
{
  if (!oneclass_)
  {
    probs.resize(nclass_,inputs.n_cols);
    for(size_t i=0;i<unclass_;i++)
    {
      arma::Row<size_t> temp;
      arma::Mat<T> tprobs;
      models_[i].Classify(inputs,temp,tprobs);
      probs.row(unq_(i)) = tprobs.row(0);
    }
    preds = unq_.cols(arma::index_min(arma::abs(probs),0));
    probs = probs.each_row() / arma::sum(probs,0);
  }
  else
  {
    preds.resize(inputs.n_cols);
    preds.fill(unq_[0]);
  }
}
template<class MODEL, class T>
T OnevAll<MODEL,T>::ComputeError ( const arma::Mat<T>& inputs,
                                   const arma::Row<size_t>& labels )
{
  arma::Row<size_t> predictions;
  Classify(inputs, predictions);
  arma::Row<size_t> temp =  predictions - labels; 
  return (arma::accu(temp != 0))/T(predictions.n_elem);
}

template<class MODEL, class T>
T OnevAll<MODEL,T>::ComputeAccuracy ( const arma::Mat<T>& inputs,
                                      const arma::Row<size_t>& labels )
{
  return 1.-ComputeError(inputs,labels);
}


} // namespace classification
} // namespace algo

#endif
