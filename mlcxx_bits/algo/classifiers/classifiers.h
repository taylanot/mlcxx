/**
 * @file classifiers.h
 * @author Ozgur Taylan Turan
 *
 */

#ifndef CLASSIFIERS_H
#define CLASSIFIERS_H

#include "multiclass.h"
#include "paramclass.h"
#include "nonparamclass.h"
#include "svm.h"
#include "wrappers.h"

namespace algo::classification
{
  //---------------------------------------------------------------------------
  // extract_class  : Get all the data from one class
  //---------------------------------------------------------------------------
  template<class T=DTYPE>
  std::tuple< arma::Mat<T>,
              arma::uvec > extract_class ( const arma::Mat<T>& inputs,
                                           const arma::Row<size_t>& labels,
                                           const size_t& label_id )
  {
    arma::uvec index = arma::find(labels == label_id);
    return std::make_tuple(inputs.cols(index), index);
  }

  //---------------------------------------------------------------------------
  // get_prior  : Estimates prior from given labels
  //---------------------------------------------------------------------------
  template<class T=DTYPE>
  arma::Row<T> get_prior ( const arma::Row<size_t>& labels,
                           const size_t& num_class )
  {
    auto unq = arma::regspace<arma::Row<size_t>>(0,1,num_class);
    auto prior = arma::conv_to<arma::Row<T>>::from(arma::hist(labels, unq));
    return  prior / labels.n_cols;
  }
}

#endif
