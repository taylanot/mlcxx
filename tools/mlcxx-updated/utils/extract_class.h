/**
 * @file extract_class.h
 * @author Ozgur Taylan Turan
 *
 * Utility for extracting a class from classification data
 *
 */

#ifndef EXTRACT_CLASS_H
#define EXTRACT_CLASS_H

namespace utils {

//arma::mat extract_class ( const arma::mat& inputs,
//                          const arma::rowvec& labels,
//                          const double& label_id )
//{
//  arma::uvec index = arma::find(labels == label_id);
//  return inputs.cols(index);
//}

std::tuple< arma::mat,
            arma::uvec > extract_class ( const arma::mat& inputs,
                                         const arma::rowvec& labels,
                                         const double& label_id )
{
  arma::uvec index = arma::find(labels == label_id);
  return std::make_tuple(inputs.cols(index), index);
}

} // namespace utils

#endif
