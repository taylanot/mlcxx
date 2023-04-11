/**
 * @file nmc_impl.h
 * @author Ozgur Taylan Turan
 *
 * Nearest Mean Classifier 
 *
 */

#ifndef NMC_IMPL_H
#define NMC_IMPL_H

namespace algo { 
namespace classification {

///////////////////////////////////////////////////////////////////////////////
// Nearest Mean Classifier 
///////////////////////////////////////////////////////////////////////////////

NMC::NMC ( const arma::mat& inputs,
           const arma::Row<size_t>& labels )
{
  Train(inputs, labels);
}

void NMC::Train ( const arma::mat& inputs,
                  const arma::Row<size_t>& labels )
{

  dim_ = inputs.n_rows;
  unique_ = arma::unique(labels);
  num_class_ = unique_.n_cols;
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
    parameters_.col(counter) = arma::mean(inputs.cols(index),1);
    counter++;
  }
}

void NMC::Classify ( const arma::mat& inputs,
                     arma::Row<size_t>& labels ) const
{
  const size_t N =  inputs.n_cols;
  labels.resize(N);
  if ( num_class_ == 1 )
    labels.fill(unique_(0));
  else
  {
    arma::mat distances(num_class_, N);
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

double NMC::ComputeError ( const arma::mat& points, 
                           const arma::Row<size_t>& responses ) const
{
  arma::Row<size_t> predictions;
  Classify(points,predictions);
  arma::Row<size_t> temp =  predictions - responses; 
  double total = responses.n_cols;
  return (arma::accu(temp != 0))/total;
}

double NMC::ComputeAccuracy ( const arma::mat& points, 
                              const arma::Row<size_t>& responses ) const
{
  return (1. - ComputeError(points, responses))*100;
}

} // namespace classification
} // namespace algo
#endif

