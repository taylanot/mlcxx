/**
 * @file fisherc_impl.h
 * @author Ozgur Taylan Turan
 *
 * Fisher Classifier 
 *
 */

#ifndef FISHERC_IMPL_H
#define FISHERC_IMPL_H

namespace algo { 
namespace classification {

///////////////////////////////////////////////////////////////////////////////
// Fisher's Linear Discriminant 
///////////////////////////////////////////////////////////////////////////////
FISHERC::FISHERC ( const arma::mat& inputs,
                   const arma::rowvec& labels ) : lambda_(1.e-5)
{
  Train(inputs, labels);
}

FISHERC::FISHERC ( const double& lambda,  
                   const arma::mat& inputs,
                   const arma::rowvec& labels ) : lambda_(lambda)
{
  Train(inputs, labels);
}

void FISHERC::Train ( const arma::mat& inputs,
                      const arma::rowvec& labels )
{
  dim_ = inputs.n_rows;
  unique_ = arma::unique(labels);
  num_class_ = unique_.n_cols;
  BOOST_ASSERT_MSG( num_class_ <= 2,
                          "Only 2 class classification is supported for now!");

  arma::mat C0, C1, inputs0, inputs1;
  arma::vec m0, m1;
  arma::uvec id0, id1;
  
  // extract class information from data
  auto extract0 = utils::extract_class(inputs, labels, 0);
  auto extract1 = utils::extract_class(inputs, labels, 1);
  inputs0 = std::get<0>(extract0); id0 = std::get<1>(extract0);
  inputs1 = std::get<0>(extract1); id1 = std::get<1>(extract1);

  m0 = arma::mean(inputs0,1); m1 = arma::mean(inputs1,1);

  // scale the data  
  inputs0.each_col() -= m0; inputs1.each_col() -= m1;
  // within scatter
  C0 = inputs0*inputs0.t(); C1 = inputs1*inputs1.t();
  // total scatter
  arma::mat C = C0 + C1 + arma::eye(arma::size(C0)) * lambda_;
  parameters_ = arma::solve(C, m1-m0);
  bias_ = arma::dot((m1+m0),parameters_)*0.5 - std::log(id1.n_rows/id0.n_rows);
}

void FISHERC::Classify ( const arma::mat& inputs,
                         arma::rowvec& labels ) const
{
  const size_t N = inputs.n_cols;
  labels.resize(N);

  if ( num_class_ == 1 )
  {
    labels.ones();
    labels *= unique_(0);
  }

  labels = parameters_.t()*inputs - bias_;

  for ( size_t i=0; i<N; i++ )
  {
    if ( labels(i) <=0 )
      labels(i) = unique_(0);
    else
      labels(i) = unique_(1);
  }
}

double FISHERC::ComputeError ( const arma::mat& points, 
                               const arma::rowvec& responses ) const
{
  arma::rowvec predictions;
  Classify(points,predictions);

  arma::rowvec temp =  predictions - responses; 
  double total = responses.n_cols;
  return (arma::accu(temp != 0.))/total;
}

double FISHERC::ComputeAccuracy ( const arma::mat& points, 
                                  const arma::rowvec& responses ) const
{
  return (1. - ComputeError(points, responses))*100;
}

} // namespace classification
} // namespace algo
#endif

