/**
 * @file nonparamclass_impl.h
 * @author Ozgur Taylan Turan
 *
 * Nonparametric Classifiers
 *
 */

#ifndef NONPARAMCLASS_IMPL_H
#define NONPARAMCLASS_IMPL_H

namespace algo { 
namespace classification {

//-----------------------------------------------------------------------------
// Nearest Neighbour Classifier
//-----------------------------------------------------------------------------
template<class T>
NNC<T>::NNC ( const arma::Mat<T>& inputs,
              const arma::Row<size_t>& labels,
              const size_t& num_class ): k_(1), nclass_(num_class)
{
  Train(inputs, labels);
}
///////////////////////////////////////////////////////////////////////////////
template<class T>
NNC<T>::NNC ( const arma::Mat<T>& inputs,
              const arma::Row<size_t>& labels,
              const size_t& num_class, 
              const size_t& k ) : k_(k), nclass_(num_class)
{
  Train(inputs, labels);
}
///////////////////////////////////////////////////////////////////////////////
template<class T>
void NNC<T>::Train ( const arma::Mat<T>& inputs,
                     const arma::Row<size_t>& labels )
{
  dim_ = inputs.n_rows;
  nuclass_ = arma::unique(labels).eval().n_elem;
  if (nuclass_ == 1)
    unique_ = arma::unique(labels);
  else
    unique_ = arma::regspace<arma::Row<size_t>>(0,nclass_-1);

  size_ = inputs.n_cols;
  inputs_ = inputs;
  labels_ = labels;
  if ( k_ > inputs.n_cols )
      k_ = inputs.n_cols-1;
}
///////////////////////////////////////////////////////////////////////////////
template<class T>
void NNC<T>::Train ( const arma::Mat<T>& inputs,
                     const arma::Row<size_t>& labels,
                     const size_t num_class )
{
  this ->nuclass_ = num_class;
  this -> Train(inputs,labels);
}
///////////////////////////////////////////////////////////////////////////////
template<class T>
void NNC<T>::Classify ( const arma::Mat<T>& inputs,
                        arma::Row<size_t>& labels ) const
{
  arma::Mat<T> temp;
  Classify(inputs, labels, temp);
}
///////////////////////////////////////////////////////////////////////////////
template<class T>
void NNC<T>::Classify ( const arma::Mat<T>& inputs,
                        arma::Row<size_t>& labels,
                        arma::Mat<T>& probs ) const
{
  const size_t N =  inputs.n_cols;
  probs.resize(nclass_,N);
  labels.resize(N);
  if ( nuclass_ == 1 )
  {
    labels.fill(unique_(0));
    probs.row(unique_(0)).fill(1);
  }
  else
  {
    // Potentially much faster implementation 
    
    mlpack::KNN knn(inputs_);
    arma::Mat<T> dist;
    arma::Mat<size_t> neig;
    arma::Mat<size_t> select;
    
    // Find the nns to all the samples
    knn.Search(inputs, k_, neig, dist);
    arma::Col<size_t> unq;
    
    
    /* #pragma omp parallel for */
    for ( size_t j=0; j<N; j++ )
    {
      // select the labels of the nns per sample
      select = labels_(arma::conv_to<arma::uvec>::from(neig.col(j)));
      // count how many classes does one sample has per sample
      auto count = arma::hist(select,unique_);
      auto ps =
             arma::conv_to<arma::Col<T>>::from(count); 
      probs.col(j) = ps / arma::accu(ps);
      // assign the maximum number of seen class
      labels(j) = unique_(count.index_max());
    }
  }
}
///////////////////////////////////////////////////////////////////////////////
template<class T>
T NNC<T>::ComputeError ( const arma::Mat<T>& points, 
                         const arma::Row<size_t>& responses ) const
{
  arma::Row<size_t> predictions;
  Classify(points,predictions);
  arma::Row<size_t> temp =  predictions - responses; 
  double total = responses.n_cols;

  return (arma::accu(temp != 0))/total;
}
///////////////////////////////////////////////////////////////////////////////
template<class T>
T NNC<T>::ComputeAccuracy ( const arma::Mat<T>& points, 
                            const arma::Row<size_t>& responses ) const
{
  return (1. - ComputeError(points, responses))*100;
}
///////////////////////////////////////////////////////////////////////////////
} // namespace classification
} // namespace algo
#endif

