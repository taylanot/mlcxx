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
MultiClass<MODEL,T>::MultiClass ( const arma::Mat<T>& inputs,
                                  const arma::Row<size_t>& labels,
                                  const std::string type,
                                  const Args&... args ) : type_(type)
{
  unq_ = arma::unique(labels).eval();
  if (unq_.n_elem == 1)
    oneclass_ = true;
  nclass_ = unq_.n_elem;
  Train(inputs, labels, args...);
}

template<class MODEL, class T>
template<class... Args>
MultiClass<MODEL,T>::MultiClass( const arma::Mat<T>& inputs,
                                 const arma::Row<size_t>& labels,
                                 const Args&... args ) : type_("SoftOnevsAll")
{ 
  unq_ = arma::unique(labels).eval();
  if (unq_.n_elem == 1)
    oneclass_ = true;
  nclass_ = unq_.n_elem;
  Train(inputs, labels, args...);
} 

template<class MODEL, class T>
template<class... Args>
void MultiClass<MODEL,T>::Train ( const arma::Mat<T>& inputs,
                                  const arma::Row<size_t>& labels,
                                  const Args&... args ) 
{ 
  if (!oneclass_)
  {
    if (type_ == "SoftOnevsAll" || type_ == "HardOnevsAll")
      OneVsAll(inputs, labels, args...);
    else
      PRINT_ERR("MultiClass::Not Implemented Yet!");
  }
} 

template<class MODEL, class T>
void MultiClass<MODEL,T>::Classify( const arma::Mat<T>& inputs,
                                    arma::Row<size_t>& preds )
{
  if (!oneclass_)
  {
    if (type_ == "HardOnevsAll")
    {
      arma::Mat<T> collect(nclass_,inputs.n_cols);
      #pragma omp parallel for
      for(size_t i=0;i<nclass_;i++)
      {
        double add = (unq_(0)!=0) ? unq_(i) :unq_(i)+1;
        arma::Row<size_t> temp;
        models_[i].Classify(inputs,temp);
        collect.row(i) = arma::conv_to<arma::Row<T>>::from(temp*add);
      }
      preds = arma::conv_to<arma::Row<size_t>>::from(
                                            arma::round(arma::mean(collect,0)));
    }
    else if (type_ == "SoftOnevsAll")
    {
      arma::Mat<T> collect(nclass_,inputs.n_cols);
      #pragma omp parallel for
      for(size_t i=0;i<nclass_;i++)
      {
        arma::Row<size_t> temp;
        arma::Mat<T> probs;
        models_[i].Classify(inputs,temp,probs);
        collect.row(i) = probs.row(0);
      }
      preds = unq_.cols(arma::index_min(arma::abs(collect),0));
    }
  }
  else
  {
    preds.resize(inputs.n_cols);
    preds.fill(unq_[0]);
  }
}

template<class MODEL, class T>
T MultiClass<MODEL,T>::ComputeError ( const arma::Mat<T>& inputs,
                                      const arma::Row<size_t>& labels )
{
  arma::Row<size_t> predictions;
  Classify(inputs,predictions);
  arma::Row<size_t> temp =  predictions - labels; 
  return (arma::accu(temp != 0))/T(predictions.n_elem);
}

template<class MODEL, class T>
T MultiClass<MODEL,T>::ComputeAccuracy ( const arma::Mat<T>& inputs,
                                         const arma::Row<size_t>& labels )
{
  return 1.-ComputeError(inputs,labels);
}

template<class MODEL, class T>
template<class... Args>
void MultiClass<MODEL,T>::OneVsAll ( const arma::Mat<T>& inputs,
                                     const arma::Row<size_t>& labels,
                                     const Args&... args ) 
{ 
  models_.resize(unq_.n_elem);
  /* #pragma omp parallel for */
  for(size_t i=0;i<nclass_;i++)
  {
    auto binlabels = arma::conv_to<arma::Row<size_t>>::from(labels==unq_(i));
    MODEL model(inputs,binlabels,args...);
    models_[i] = model;
  }
}


} // namespace classification
} // namespace algo

#endif
