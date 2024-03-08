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

//=============================================================================
// Parzen Classifier
//=============================================================================

PARZENC::PARZENC ( const arma::mat& inputs,
                   const arma::Row<size_t>& labels )
{
  Train(inputs, labels);
}

void PARZENC::Train ( const arma::mat& inputs,
                      const arma::Row<size_t>& labels )
{

  dim_ = inputs.n_rows;
  unique_ = arma::unique(labels);
  num_class_ = unique_.n_cols;
  size_ = inputs.n_cols;
  parameters_ = inputs;
}

void PARZENC::Classify ( const arma::mat& inputs,
                         arma::Row<size_t>& labels ) const
{

}

double PARZENC::ComputeError ( const arma::mat& points, 
                               const arma::Row<size_t>& responses ) const
{
  arma::Row<size_t> predictions;
  Classify(points,predictions);
  arma::Row<size_t> temp =  predictions - responses; 
  double total = responses.n_cols;
  return (arma::accu(temp != 0))/total;
}

double PARZENC::ComputeAccuracy ( const arma::mat& points, 
                                  const arma::Row<size_t>& responses ) const
{
  return (1. - ComputeError(points, responses))*100;
}

//=============================================================================
// Nearest Neighbour Classifier
//=============================================================================

NNC::NNC ( const arma::mat& inputs,
           const arma::Row<size_t>& labels ): k_(1)
{
  
  Train(inputs, labels);
}

NNC::NNC ( const arma::mat& inputs,
           const arma::Row<size_t>& labels,
           const size_t& k ) : k_(k)
{
  
  Train(inputs, labels);
}

void NNC::Train ( const arma::mat& inputs,
                  const arma::Row<size_t>& labels )
{
  dim_ = inputs.n_rows;
  unique_ = arma::unique(labels);
  num_class_ = unique_.n_cols;
  size_ = inputs.n_cols;
  inputs_ = inputs;
  labels_ = labels;
  if ( k_ > inputs.n_cols )
      k_ = inputs.n_cols-1;
}

void NNC::Classify ( const arma::mat& inputs,
                     arma::Row<size_t>& labels ) const
{
 const size_t N =  inputs.n_cols;
  labels.resize(N);
  if ( num_class_ == 1 )
    labels.fill(unique_(0));
  else
  {
    // Potentially much faster implementation 
    
    mlpack::KNN knn(inputs_);
    arma::mat dist;
    arma::Mat<size_t> neig;
    arma::Mat<size_t> select;

    knn.Search(inputs, k_, neig, dist);
    arma::Col<size_t> unq;
    
    //#pragma omp parallel for
    for ( size_t j=0; j<N; j++ )
    {
      select = labels_(arma::conv_to<arma::uvec>::from(neig.col(j)));
      unq = arma::unique(select);
      auto count =
             arma::conv_to<arma::Col<size_t>>::from(arma::hist(select,unq));  
      labels(j) = unq(count.index_max());
    }
    ///PRINT_VAR(neig_lab);
    //auto count =
    //         arma::conv_to<arma::Col<size_t>>::from(arma::hist(select,unq));
    //arma::mat lab
    //knn.search(inputs,k_,neigh,dist);
  }
}

double NNC::ComputeError ( const arma::mat& points, 
                           const arma::Row<size_t>& responses ) const
{
  arma::Row<size_t> predictions;
  Classify(points,predictions);
  arma::Row<size_t> temp =  predictions - responses; 
  double total = responses.n_cols;

  return (arma::accu(temp != 0))/total;
}

double NNC::ComputeAccuracy ( const arma::mat& points, 
                              const arma::Row<size_t>& responses ) const
{
  return (1. - ComputeError(points, responses))*100;
}

} // namespace classification
} // namespace algo
#endif

