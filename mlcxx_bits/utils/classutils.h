/**
 * @file extract_class.h
 * @author Ozgur Taylan Turan
 *
 * Utility for classification
 *
 */

#ifndef CLASSUTILS_H
#define CLASSUTILS_H

namespace utils {

//-----------------------------------------------------------------------------
// extract_class : Get all the data from one class
//-----------------------------------------------------------------------------
std::tuple< arma::mat,
            arma::uvec > extract_class ( const arma::mat& inputs,
                                         const arma::Row<size_t>& labels,
                                         const size_t& label_id )
{
  arma::uvec index = arma::find(labels == label_id);
  return std::make_tuple(inputs.cols(index), index);
}

//-----------------------------------------------------------------------------
// GetPrior : Estimates prior from given labels
//-----------------------------------------------------------------------------

arma::rowvec GetPrior ( const arma::Row<size_t>& labels )
{
  arma::Row<size_t> unq = arma::unique(labels);
  auto prior = arma::conv_to<arma::rowvec>::from(arma::hist(labels, unq));
  return  prior / labels.n_cols;
  
}
} // namespace utils

#endif
