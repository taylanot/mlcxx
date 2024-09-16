/**
 * @file paramclass_impl.h
 * @author Ozgur Taylan Turan
 *
 * Parametric Classifiers
 *
 */

#ifndef PARAMCLASS_IMPL_H
#define PARAMCLASS_IMPL_H

namespace algo { 
namespace classification {

//=============================================================================
// Linear Discriminant Classifier 
//=============================================================================

template<class T>
LDC<T>::LDC ( const arma::Mat<T>& inputs,
              const arma::Row<size_t>& labels,
              const size_t& num_class ) :  num_class_(num_class), lambda_(0.)
{
  class_ = arma::regspace<arma::Row<size_t>>(0,1,num_class_);
  priors_ = utils::GetPrior(labels);
  Train(inputs, labels);
}

template<class T>
LDC<T>::LDC ( const arma::Mat<T>& inputs,
              const arma::Row<size_t>& labels,
              const size_t& num_class,
              const double& lambda,
              const arma::Row<T>& priors ) : num_class_(num_class),
                                             lambda_(lambda), priors_(priors)
{
  class_ = arma::regspace<arma::Row<size_t>>(0,1,num_class_);
  priors_ = utils::GetPrior(labels);
  Train(inputs, labels);
}

template<class T>
LDC<T>::LDC ( const arma::Mat<T>& inputs,
              const arma::Row<size_t>& labels,
              const size_t& num_class,
              const double& lambda ) : num_class_(num_class), lambda_(lambda)
{
  class_ = arma::regspace<arma::Row<size_t>>(0,1,num_class_);
  priors_ = utils::GetPrior(labels);
  Train(inputs, labels);
}

template<class T>
void LDC<T>::Train ( const arma::Mat<T>& inputs,
                     const arma::Row<size_t>& labels )
{
  dim_ = inputs.n_rows;
  size_ = inputs.n_cols;
  unique_ = arma::unique(labels);
  if (unique_.n_elem != 1)
  {
    cov_.resize(dim_,dim_);
    cov_.zeros();
    
    arma::Row<size_t>::iterator it = unique_.begin();
    arma::Row<size_t>::iterator end = unique_.end();

    arma::Mat<T> inx;

    for(; it!=end; it++)
    {
      auto extract = utils::extract_class(inputs, labels, *it);
      inx = std::get<0>(extract);
      means_[*it] = arma::conv_to<arma::Row<T>>::from(arma::mean(inx,1));
      if ( inx.n_cols == 1 )
      {
        covs_[*it] = arma::eye<arma::Mat<T>>(dim_,dim_);
      }
      else
      {
        covs_[*it] = arma::cov(inx.t());
        covs_[*it].diag() += jitter_+lambda_;
      }
      cov_ += covs_[*it];
    }
    cov_ = cov_.i() / num_class_;
  }
}

template<class T>
void LDC<T>::Classify ( const arma::Mat<T>& inputs,
                        arma::Row<size_t>& labels ) const
{
  arma::Mat<T> temp;
  Classify(inputs, labels, temp);
}

template<class T>
void LDC<T>::Classify ( const arma::Mat<T>& inputs,
                        arma::Row<size_t>& labels,
                        arma::Mat<T>& probs ) const
{
  const size_t N = inputs.n_cols;
  labels.resize(N);
  probs.resize(num_class_,N);
  if ( unique_.n_elem == 1 )
  {
    labels.fill(unique_(0));
    probs.row(unique_(0)).fill(1.);
  }
  else
  {
    for ( size_t n=0; n<inputs.n_cols; n++ ) 
    {
      for ( size_t c=0; c<unique_.n_elem; c++ )
      {
        probs(class_(unique_(c)),n) = std::log(priors_(c)) 
                       -  0.5*arma::dot(means_.at(unique_(c))*
                        cov_, means_.at(unique_(c)))
                    + arma::dot(inputs.col(n).t()*cov_,means_.at(unique_(c)));
      }
      labels(n) = class_(probs.col(n).index_max());
      
    }
    probs = arma::exp(probs.each_row() - arma::max(probs,0));
    probs = probs.each_row()/arma::sum(probs,0);
  }
}

template<class T>
T LDC<T>::ComputeError ( const arma::Mat<T>& points, 
                         const arma::Row<size_t>& responses ) const
{
  arma::Row<size_t> predictions;
  Classify(points,predictions);
  arma::Row<size_t> temp =  predictions - responses; 
  return (arma::accu(temp != 0))/T(predictions.n_elem);
}

template<class T>
T LDC<T>::ComputeAccuracy ( const arma::Mat<T>& points, 
                            const arma::Row<size_t>& responses ) const
{
  return (1. - ComputeError(points, responses))*100;
}

//=============================================================================
// Quadratic Discriminant Classifier 
//=============================================================================

template<class T>
QDC<T>::QDC ( const arma::Mat<T>& inputs,
              const arma::Row<size_t>& labels,
              const size_t& num_class ) : num_class_(num_class), lambda_(0.)
{
  class_ = arma::regspace<arma::Row<size_t>>(0,1,num_class_);
  priors_ = utils::GetPrior(labels);
  Train(inputs, labels);
}

template<class T>
QDC<T>::QDC ( const arma::Mat<T>& inputs,
              const arma::Row<size_t>& labels,
              const size_t& num_class,
              const double& lambda,
              const arma::Row<T>& priors ) : num_class_(num_class),
                                             lambda_(lambda), priors_(priors)
{
  class_ = arma::regspace<arma::Row<size_t>>(0,1,num_class_);
  priors_ = utils::GetPrior(labels);
  Train(inputs, labels);
}

template<class T>
QDC<T>::QDC ( const arma::Mat<T>& inputs,
              const arma::Row<size_t>& labels,
              const size_t& num_class,
              const double& lambda ) : num_class_(num_class), lambda_(lambda)
{
  class_ = arma::regspace<arma::Row<size_t>>(0,1,num_class_);
  priors_ = utils::GetPrior(labels);
  Train(inputs, labels);
}

template<class T>
void QDC<T>::Train ( const arma::Mat<T>& inputs,
                     const arma::Row<size_t>& labels )
{
  dim_ = inputs.n_rows;
  size_ = inputs.n_cols;
  unique_ = arma::unique(labels);
  
  arma::Row<size_t>::iterator it = unique_.begin();
  arma::Row<size_t>::iterator end = unique_.end();

  arma::Mat<T> inx;

  for(; it!=end; it++)
  {
    auto extract = utils::extract_class(inputs, labels, *it);

    inx = std::get<0>(extract);
    means_[*it] = arma::conv_to<arma::Row<T>>::from(arma::mean(inx,1));
    if ( inx.n_cols == 1 )
    {
      covs_[*it] = arma::eye<arma::Mat<T>>(dim_,dim_);
      icovs_[*it] = arma::eye<arma::Mat<T>>(dim_,dim_);
    }
    else
    {
      covs_[*it] = arma::cov(inx.t());
      covs_[*it].diag() += jitter_+lambda_;
      icovs_[*it] = covs_[*it].i();
    }
    icovs_[*it] = covs_[*it].i();
  }
}

template<class T>
void QDC<T>::Classify ( const arma::Mat<T>& inputs,
                        arma::Row<size_t>& labels ) const
{
  arma::Mat<T> temp;
  Classify(inputs, labels, temp);
}

template<class T>
void QDC<T>::Classify ( const arma::Mat<T>& inputs,
                        arma::Row<size_t>& labels,
                        arma::Mat<T>& probs ) const
{
  const size_t N = inputs.n_cols;
  labels.resize(N);
  probs.resize(num_class_,N);

  if ( num_class_ == 1 )
  {
    labels.fill(unique_(0));
    probs.row(unique_(0)).fill(1.);
  }
  else
  {
    arma::Row<T> norm;

    for ( size_t n=0; n<inputs.n_cols; n++ ) 
    {
      for ( size_t c=0; c<unique_.n_elem; c++ )
      {
        norm = inputs.col(n).t() - means_.at(unique_(c));
        probs(class_(unique_(c)),n) = std::log(priors_(c)) 
                  -  0.5*arma::det(covs_.at(unique_(c)))
                  - 0.5* arma::dot(norm*icovs_.at(unique_(c)),norm);
      }
      labels(n) = class_(probs.col(n).index_max());
      
    }
    probs = arma::exp(probs.each_row() - arma::max(probs,0));
    probs = probs.each_row()/arma::sum(probs,0);
  }
}

template<class T>
T QDC<T>::ComputeError ( const arma::Mat<T>& points, 
                         const arma::Row<size_t>& responses ) const
{
  arma::Row<size_t> predictions;
  Classify(points,predictions);
  arma::Row<size_t> temp =  predictions - responses; 
  return (arma::accu(temp != 0))/T(predictions.n_elem);
}

template<class T>
T QDC<T>::ComputeAccuracy ( const arma::Mat<T>& points, 
                            const arma::Row<size_t>& responses ) const
{
  return (1. - ComputeError(points, responses))*100;
}

//=============================================================================
// Fisher's Linear Discriminant 
//=============================================================================

template<class T>
FDC<T>::FDC ( const arma::Mat<T>& inputs,
              const arma::Row<size_t>& labels ) : lambda_(1.e-5)
{
  Train(inputs, labels);
}

template<class T>
FDC<T>::FDC ( const arma::Mat<T>& inputs,
              const arma::Row<size_t>& labels,
              const double& lambda ) : lambda_(lambda)
{
  Train(inputs, labels);
}

template<class T>
void FDC<T>::Train ( const arma::Mat<T>& inputs,
                     const arma::Row<size_t>& labels )
{
  dim_ = inputs.n_rows;
  unique_ = arma::unique(labels);
  num_class_ = unique_.n_cols;
  BOOST_ASSERT_MSG( num_class_ <= 2,
                     "Only up to2 class classification is supported for now!");

  arma::Mat<T> C0, C1, inputs0, inputs1;
  arma::Col<T> m0, m1;
  arma::uvec id0, id1;
  
  if ( num_class_ == 2 )
  {
    // extract class information from data
    auto extract0 = utils::extract_class(inputs, labels, unique_(0));
    auto extract1 = utils::extract_class(inputs, labels, unique_(1));

    inputs0 = std::get<0>(extract0); id0 = std::get<1>(extract0);
    inputs1 = std::get<0>(extract1); id1 = std::get<1>(extract1);

    m0 = arma::mean(inputs0,1); m1 = arma::mean(inputs1,1);

    // scale the data  
    inputs0.each_col() -= m0; inputs1.each_col() -= m1;
    // within scatter
    C0 = inputs0*inputs0.t(); C1 = inputs1*inputs1.t();
    // total scatter
    arma::Mat<T> C = C0 + C1 + arma::eye<arma::Mat<T>>(arma::size(C0)) * lambda_;
    parameters_ = arma::solve(C, m1-m0);
    bias_ = arma::dot((m1+m0),parameters_)*0.5 -
                                               std::log(id1.n_rows/id0.n_rows);
  }
}

template<class T>
void FDC<T>::Classify ( const arma::Mat<T>& inputs,
                        arma::Row<size_t>& labels ) const
{
  const size_t N = inputs.n_cols;
  labels.resize(N);
  arma::Row<T> temp_labels;
  if ( num_class_ == 1 )
    labels.fill(unique_(0));
  else
  {
    temp_labels = parameters_.t()*inputs - bias_;

    for ( size_t i=0; i<N; i++ )
    {
      if ( temp_labels(i) <=0 )
        labels(i) = unique_(0);
      else
        labels(i) = unique_(1);
    }
  }
}

template<class T>
T FDC<T>::ComputeError ( const arma::Mat<T>& points, 
                         const arma::Row<size_t>& responses ) const
{
  arma::Row<size_t> predictions;
  Classify(points,predictions);
  arma::Row<size_t> temp =  predictions - responses; 
  return (arma::accu(temp != 0))/T(predictions.n_elem);
}

template<class T>
T FDC<T>::ComputeAccuracy ( const arma::Mat<T>& points, 
                            const arma::Row<size_t>& responses ) const
{
  return (1. - ComputeError(points, responses))*100;
}

//=============================================================================
// Nearest Mean Classifier 
//=============================================================================
template<class T>
NMC<T>::NMC ( const arma::Mat<T>& inputs,
              const arma::Row<size_t>& labels,
              const double& shrink ) : shrink_(shrink)
{
  Train(inputs, labels);
}

template<class T>
NMC<T>::NMC ( const arma::Mat<T>& inputs,
              const arma::Row<size_t>& labels ) : shrink_(0.)
{
  Train(inputs, labels);
}

template<class T>
void NMC<T>::Train ( const arma::Mat<T>& inputs,
                     const arma::Row<size_t>& labels )
{

  dim_ = inputs.n_rows;
  unique_ = arma::unique(labels);
  num_class_ = unique_.n_cols;
  size_ =  inputs.n_cols;
  arma::vec nk(num_class_);
  parameters_.resize(inputs.n_rows, num_class_);
  arma::uvec index;
  arma::Row<size_t>::iterator it = unique_.begin();
  arma::Row<size_t>::iterator it_end = unique_.end();
  size_t counter =0;
  // you should just iterate over the unique labels here instead of starting 
  // from 0

  for ( ;it!=it_end; ++it)
  {
    auto extract = utils::extract_class(inputs, labels, *it);
    //index = arma::find(labels == unique_(i));
    index = std::get<1>(extract);
    nk(counter) = index.n_rows;
    parameters_.col(counter) = arma::mean(inputs.cols(index),1);
    counter++;
  }
  centroid_ = arma::mean(inputs, 1); 
  
  // Just for the shrinkage part 
  if (shrink_ > 0 && num_class_ != 1) 
  {
    arma::Mat<T> nk = arma::ones<arma::Mat<T>>(num_class_,1);
    arma::Mat<T> m = arma::sqrt(1./nk)-1/size_;
    arma::uvec labs = arma::conv_to<arma::uvec>::from(labels);
    arma::Mat<T> variance = arma::sum(arma::pow(inputs - parameters_.cols(labs),2),1);
    arma::Mat<T> s = arma::sqrt(variance/(size_-num_class_)).t();
    arma::Mat<T> ms = m*s;
    arma::inplace_trans(ms);
    arma::Mat<T> devi = (parameters_.each_col() - centroid_) / ms;
    arma::Mat<T> signs = arma::sign(devi);
    arma::Mat<T> dev = arma::abs(devi) - shrink_;
    dev = arma::clamp(dev, 0, arma::datum::inf);
    dev  %= signs; 
    arma::Mat<T> msd = dev % ms;
    parameters_ = centroid_ + msd.each_col();
  }
}

template<class T>
void NMC<T>::Classify ( const arma::Mat<T>& inputs,
                        arma::Row<size_t>& labels ) const
{

  const size_t N =  inputs.n_cols;
  labels.resize(N);
  if ( num_class_ == 1 )
    labels.fill(unique_(0));
  else
  {
    arma::Mat<T> distances(num_class_, N);
    arma::urowvec index(N);
    for ( size_t j=0; j<N; j++ )
    {
      for ( size_t i=0; i<num_class_; i++ )
      { 
        distances(i, j) = metric_.Evaluate(parameters_.col(i),inputs.col(j));
      }
        index(j) = arma::index_min(distances.col(j));
        labels(j) = unique_(index(j));

    }
  }
 
}

template<class T>
T NMC<T>::ComputeError ( const arma::Mat<T>& points, 
                         const arma::Row<size_t>& responses ) const
{
  arma::Row<size_t> predictions;
  
  Classify(points,predictions);
  arma::Row<size_t> temp =  predictions - responses; 
  return (arma::accu(temp != 0))/T(predictions.n_elem);
}

template<class T>
T NMC<T>::ComputeAccuracy ( const arma::Mat<T>& points, 
                            const arma::Row<size_t>& responses ) const
{
  return (1. - ComputeError(points, responses))*100;
}
} // namespace classification
} // namespace algo
#endif

