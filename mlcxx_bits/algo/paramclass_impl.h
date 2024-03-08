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

LDC::LDC ( const arma::mat& inputs,
           const arma::Row<size_t>& labels ) : lambda_(0.)
{
  priors_ = utils::GetPrior(labels);
  Train(inputs, labels);
}

LDC::LDC ( const arma::mat& inputs,
           const arma::Row<size_t>& labels,
           const double& lambda,
           const arma::rowvec& priors ) : lambda_(lambda), priors_(priors)
{
  priors_ = utils::GetPrior(labels);
  Train(inputs, labels);
}

LDC::LDC ( const arma::mat& inputs,
           const arma::Row<size_t>& labels,
           const double& lambda ) : lambda_(lambda)
{
  priors_ = utils::GetPrior(labels);
  Train(inputs, labels);
}

void LDC::Train ( const arma::mat& inputs,
                  const arma::Row<size_t>& labels )
{
  dim_ = inputs.n_rows;
  size_ = inputs.n_cols;
  unique_ = arma::unique(labels);
  num_class_ = unique_.n_cols;
  cov_.resize(dim_,dim_);
  cov_.zeros();
  
  arma::Row<size_t>::iterator it = unique_.begin();
  arma::Row<size_t>::iterator end = unique_.end();

  arma::mat inx;

  for(; it!=end; it++)
  {
    auto extract = utils::extract_class(inputs, labels, *it);

    inx = std::get<0>(extract);
    
    means_[*it] = arma::conv_to<arma::rowvec>::from(arma::mean(inx,1));
    if ( inx.n_cols == 1 )
    {
      covs_[*it] = arma::eye(dim_,dim_);
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

void LDC::Classify ( const arma::mat& inputs,
                     arma::Row<size_t>& labels ) const
{
  const size_t N = inputs.n_cols;
  labels.resize(N);
  arma::rowvec temp_labels;

  if ( num_class_ == 1 )
    labels.fill(unique_(0));
  else
  {
    std::map<size_t, arma::rowvec> scores;

    arma::rowvec class_scores;
    for ( size_t n=0; n<inputs.n_cols; n++ ) 
    {
      class_scores.resize(num_class_);
      for ( size_t c=0; c<num_class_; c++ )
      {
        class_scores(c) = std::log(priors_(c)) 
                  -  0.5*arma::dot(means_.at(unique_(c))*
                      cov_, means_.at(unique_(c)))
                  + arma::dot(inputs.col(n).t()*cov_,means_.at(unique_(c)));
      }
        labels(n) = unique_(class_scores.index_max());
    }
  }
}

double LDC::ComputeError ( const arma::mat& points, 
                           const arma::Row<size_t>& responses ) const
{
  arma::Row<size_t> predictions;
  Classify(points,predictions);
  arma::Row<size_t> temp =  predictions - responses; 
  double total = responses.n_cols;
  return (arma::accu(temp != 0))/total;
}

double LDC::ComputeAccuracy ( const arma::mat& points, 
                              const arma::Row<size_t>& responses ) const
{
  return (1. - ComputeError(points, responses))*100;
}

//=============================================================================
// Quadratic Discriminant Classifier 
//=============================================================================

QDC::QDC ( const arma::mat& inputs,
           const arma::Row<size_t>& labels ) : lambda_(0.)
{
  priors_ = utils::GetPrior(labels);
  Train(inputs, labels);
}

QDC::QDC ( const arma::mat& inputs,
           const arma::Row<size_t>& labels,
           const double& lambda,
           const arma::rowvec& priors ) : lambda_(lambda), priors_(priors)
{
  priors_ = utils::GetPrior(labels);
  Train(inputs, labels);
}

QDC::QDC ( const arma::mat& inputs,
           const arma::Row<size_t>& labels,
           const double& lambda ) : lambda_(lambda)
{
  priors_ = utils::GetPrior(labels);
  Train(inputs, labels);
}

void QDC::Train ( const arma::mat& inputs,
                  const arma::Row<size_t>& labels )
{
  dim_ = inputs.n_rows;
  size_ = inputs.n_cols;
  unique_ = arma::unique(labels);
  num_class_ = unique_.n_cols;
  
  arma::Row<size_t>::iterator it = unique_.begin();
  arma::Row<size_t>::iterator end = unique_.end();

  arma::mat inx;

  for(; it!=end; it++)
  {
    auto extract = utils::extract_class(inputs, labels, *it);

    inx = std::get<0>(extract);
    means_[*it] = arma::conv_to<arma::rowvec>::from(arma::mean(inx,1));
    if ( inx.n_cols == 1 )
    {
      covs_[*it] = arma::eye(dim_,dim_);
      icovs_[*it] = arma::eye(dim_,dim_);
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

void QDC::Classify ( const arma::mat& inputs,
                     arma::Row<size_t>& labels ) const
{
  const size_t N = inputs.n_cols;
  labels.resize(N);
  arma::rowvec temp_labels;

  if ( num_class_ == 1 )
    labels.fill(unique_(0));
  else
  {
    std::map<size_t, arma::rowvec> scores;

    arma::rowvec class_scores;
    arma::rowvec norm;
    for ( size_t n=0; n<inputs.n_cols; n++ ) 
    {
      class_scores.resize(num_class_);
      for ( size_t c=0; c<num_class_; c++ )
      {
        norm = inputs.col(n).t() - means_.at(unique_(c));
        class_scores(c) = std::log(priors_(c)) 
                  -  0.5*arma::det(covs_.at(unique_(c)))
                  - 0.5* arma::dot(norm*icovs_.at(unique_(c)),norm);
      }
        labels(n) = unique_(class_scores.index_max());
    }
   
  }
}

double QDC::ComputeError ( const arma::mat& points, 
                           const arma::Row<size_t>& responses ) const
{
  arma::Row<size_t> predictions;
  Classify(points,predictions);
  arma::Row<size_t> temp =  predictions - responses; 
  double total = responses.n_cols;
  return (arma::accu(temp != 0))/total;
}

double QDC::ComputeAccuracy ( const arma::mat& points, 
                              const arma::Row<size_t>& responses ) const
{
  return (1. - ComputeError(points, responses))*100;
}

//=============================================================================
// Fisher's Linear Discriminant 
//=============================================================================

FDC::FDC ( const arma::mat& inputs,
           const arma::Row<size_t>& labels ) : lambda_(1.e-5)
{
  Train(inputs, labels);
}

FDC::FDC ( const arma::mat& inputs,
           const arma::Row<size_t>& labels,
           const double& lambda ) : lambda_(lambda)
{
  Train(inputs, labels);
}

void FDC::Train ( const arma::mat& inputs,
                  const arma::Row<size_t>& labels )
{
  dim_ = inputs.n_rows;
  unique_ = arma::unique(labels);
  num_class_ = unique_.n_cols;
  BOOST_ASSERT_MSG( num_class_ <= 2,
                     "Only up to2 class classification is supported for now!");

  arma::mat C0, C1, inputs0, inputs1;
  arma::vec m0, m1;
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
    arma::mat C = C0 + C1 + arma::eye(arma::size(C0)) * lambda_;
    parameters_ = arma::solve(C, m1-m0);
    bias_ = arma::dot((m1+m0),parameters_)*0.5 -
                                               std::log(id1.n_rows/id0.n_rows);
  }
}

void FDC::Classify ( const arma::mat& inputs,
                         arma::Row<size_t>& labels ) const
{
  const size_t N = inputs.n_cols;
  labels.resize(N);
  arma::rowvec temp_labels;
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

double FDC::ComputeError ( const arma::mat& points, 
                               const arma::Row<size_t>& responses ) const
{
  arma::Row<size_t> predictions;
  Classify(points,predictions);
  arma::Row<size_t> temp =  predictions - responses; 
  double total = responses.n_cols;
  return (arma::accu(temp != 0))/total;
}

double FDC::ComputeAccuracy ( const arma::mat& points, 
                                  const arma::Row<size_t>& responses ) const
{
  return (1. - ComputeError(points, responses))*100;
}

//=============================================================================
// Nearest Mean Classifier 
//=============================================================================

NMC::NMC ( const arma::mat& inputs,
           const arma::Row<size_t>& labels,
           const double& shrink ) : shrink_(shrink)
{
  Train(inputs, labels);
}

NMC::NMC ( const arma::mat& inputs,
           const arma::Row<size_t>& labels ) : shrink_(0.)
{
  Train(inputs, labels);
}

void NMC::Train ( const arma::mat& inputs,
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
    arma::mat nk = arma::ones(num_class_,1);
    arma::mat m = arma::sqrt(1./nk)-1/size_;
    arma::uvec labs = arma::conv_to<arma::uvec>::from(labels);
    arma::mat variance = arma::sum(
                              arma::pow(inputs - parameters_.cols(labs),2),1);
    arma::mat s = arma::sqrt(variance/(size_-num_class_)).t();
    arma::mat ms = m*s;
    arma::inplace_trans(ms);
    arma::mat devi = (parameters_.each_col() - centroid_) / ms;
    arma::mat signs = arma::sign(devi);
    arma::mat dev = arma::abs(devi) - shrink_;
    dev = arma::clamp(dev, 0, arma::datum::inf);
    dev  %= signs; 
    arma::mat msd = dev % ms;
    parameters_ = centroid_ + msd.each_col();
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

