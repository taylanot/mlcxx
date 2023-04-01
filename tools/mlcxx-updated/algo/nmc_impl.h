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
           const arma::rowvec& labels )
{
  Train(inputs, labels);
}

void NMC::Train ( const arma::mat& inputs,
                  const arma::rowvec& labels )
{
  dim_ = inputs.n_rows;
  unique_ = arma::unique(labels);
  num_class_ = unique_.n_cols;
  parameters_.resize(inputs.n_rows, num_class_);
  arma::uvec index;
  for ( size_t i=0; i<num_class_; i++ )
  {
    auto extract = utils::extract_class(inputs, labels, i);
    //index = arma::find(labels == unique_(i));
    index = std::get<1>(extract);
    parameters_.col(i) = arma::mean(inputs.cols(index),1);
  }
}

void NMC::Classify ( const arma::mat& inputs,
                     arma::rowvec& labels ) const
{

  const size_t N =  inputs.n_cols;
  labels.resize(N);
  if ( num_class_ == 1 )
  {
    labels.ones();
    labels *= unique_(0);
  }
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
                           const arma::rowvec& responses ) const
{
  arma::rowvec predictions;
  Classify(points,predictions);

  arma::rowvec temp =  predictions - responses; 
  double total = responses.n_cols;
  return (arma::accu(temp != 0.))/total;
}

double NMC::ComputeAccuracy ( const arma::mat& points, 
                              const arma::rowvec& responses ) const
{
  return (1. - ComputeError(points, responses))*100;
}

} // namespace classification
} // namespace algo
#endif

