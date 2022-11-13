/**
 * @file learning_curve.h
 * @author Ozgur Taylan Turan
 *
 * Learning curve generator for an MLALgorithm given a dataset 
 *
 * TODO: 
 * - Add training errors.
 * - Write it much cleaner.
 *
 *
 */

#ifndef LEARNING_CURVE_H
#define LEARNING_CURVE_H

// standard
#include <tuple>
#include <iostream>
#include <omp.h>
#include <string>
// mlpack
#include <mlpack/core/cv/simple_cv.hpp>
// boost
#include <boost/assert.hpp>
// local
#include <utils/split_data.h>
#include <utils/save.h>

//=============================================================================
// LCurve_HPT 
//=============================================================================
template<class MODEL,
         class LOSS, 
         template<typename, typename, typename, typename, typename> class CV>
class LCurve_HPT
{
public:
  LCurve_HPT<MODEL,LOSS,CV>(const std::string filename,
                                   const arma::irowvec& Ns,
                                   const double& repeat,
                                   const double& tune_ratio);

  template<class... Ts>
  std::tuple<arma::mat, arma::mat> Generate(const arma::mat& inputs,
                                            const arma::rowvec& labels,
                                            const Ts&... args);
private:
  double tune_ratio_;
  size_t repeat_;
  arma::irowvec Ns_;
  std::string filename_;
  arma::mat test_errors_;
  arma::mat train_errors_;
};

//=============================================================================
// LCurve 
//=============================================================================
template<class MODEL,
         class LOSS>
class LCurve
{
public:
  LCurve<MODEL,LOSS>(const std::string filename,
                            const arma::irowvec& Ns,
                            const double& repeat);

  template<class... Ts>
  std::tuple<arma::mat, arma::mat> Generate(const arma::mat& inputs,
                                            const arma::rowvec& labels,
                                            const Ts&... args);
private:

  size_t repeat_;
  arma::irowvec Ns_;
  std::string filename_;
  arma::mat test_errors_;
  arma::mat train_errors_;
};
#include "learning_curve_impl.h"

#endif

