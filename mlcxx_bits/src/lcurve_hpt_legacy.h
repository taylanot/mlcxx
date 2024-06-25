/**
 * @file lcurve_hptlegecy.h
 * @author Ozgur Taylan Turan
 *
 * Learning curve generator for an MLALgorithm given a dataset with 
 *  hyper parameter optimization
 *
 * TODO: 
 *
 */

#ifndef LCURVE_HPT_LEGACY_H
#define LCURVE_HPT_LEGACY_H

namespace src {
namespace regression {

//=============================================================================
// LCurve_HPT 
//=============================================================================

template<class MODEL,
         class LOSS, 
         template<typename, typename, typename, typename, typename> class CV>
class LCurve_HPT
{
public:
  /* Learning Curve with inner CV for hyper-parameter tunning
   *
   * @param Ns          : row vector of training points 
   * @param repeat      : amount of times the training for single N takes place
   * @param tune_ratio  : how much of the data is used for tuning
   *
   */
  LCurve_HPT<MODEL,LOSS,CV> ( const arma::irowvec& Ns,
                              const double& repeat,
                              const double& tune_ratio );

  /* Generate Learning Curve and save statistics to a file
   *
   * @param filename  : filename with extension to save
   * @param inputs    : whole large dataset inputs
   * @param labels    : whole large dataset labels
   * @param args      : hyper-parameter ranges 
   *
   */
  template<class... Ts>
  void Generate ( const std::filesystem::path filename,
                  const arma::mat& inputs,
                  const arma::rowvec& labels,
                  const Ts&... args );
  template<class... Ts>
  void Generate ( const std::filesystem::path filename,
                  const arma::mat& inputs,
                  const arma::mat& labels,
                  const Ts&... args );

  /* Generate Learning Curves 
   *
   * @param filename  : filename with extension to save
   * @param inputs    : whole large dataset inputs
   * @param labels    : whole large dataset labels
   * @param args      : possible arguments for the model initialization
   *
   */
  template<class... Ts>
  void Generate ( const arma::mat& inputs,
                  const arma::rowvec& labels,
                  const Ts&... args );

  template<class... Ts>
  void Generate ( const arma::mat& inputs,
                  const arma::mat& labels,
                  const Ts&... args );


  /* Save the stats to a file
   *
   * @param filename  : filename with extension to save
   *
   */
  template<class... Ts>
  void Save ( const std::filesystem::path filename);

  /* 
   *
   * @param filename  : filename with extension to save
   *
   */

  private:
  arma::irowvec Ns_;
  size_t repeat_;
  double tune_ratio_;
  arma::mat results_;


public:
  arma::mat test_errors_;
  arma::mat train_errors_;
  std::tuple<arma::mat, arma::mat>  stats_;
};

} // namespace regression
} // namespace src

namespace src {
namespace classification {

//=============================================================================
// LCurve_HPT 
//=============================================================================

template<class MODEL,
         class LOSS, 
         template<typename, typename, typename, typename, typename> class CV>
class LCurve_HPT
{
public:
  /* Learning Curve with inner CV for hyper-parameter tunning
   *
   * @param Ns          : row vector of training points 
   * @param repeat      : amount of times the training for single N takes place
   * @param tune_ratio  : how much of the data is used for tuning
   *
   */
  LCurve_HPT<MODEL,LOSS,CV> ( const arma::irowvec& Ns,
                              const double& repeat,
                              const double& tune_ratio );

  /* Generate Learning Curve and save statistics to a file
   *
   * @param filename  : filename with extension to save
   * @param inputs    : whole large dataset inputs
   * @param labels    : whole large dataset labels
   * @param args      : hyper-parameter ranges 
   *
   */
  template<class... Ts>
  void Generate ( const std::filesystem::path filename,
                  const arma::mat& inputs,
                  const arma::Row<size_t>& labels,
                  const Ts&... args );

  /* Generate Learning Curves 
   *
   * @param filename  : filename with extension to save
   * @param inputs    : whole large dataset inputs
   * @param labels    : whole large dataset labels
   * @param args      : possible arguments for the model initialization
   *
   */
  template<class... Ts>
  void Generate ( const arma::mat& inputs,
                  const arma::Row<size_t>& labels,
                  const Ts&... args );

  template<class... Ts>
  void StratifiedGenerate ( const arma::mat& inputs,
                            const arma::Row<size_t>& labels,
                            const Ts&... args );

  /* Save the stats to a file
   *
   * @param filename  : filename with extension to save
   *
   */
  template<class... Ts>
  void Save ( const std::filesystem::path filename);

private:
  arma::irowvec Ns_;
  size_t repeat_;
  double tune_ratio_;
  arma::mat results_;

public:
  arma::mat test_errors_;
  arma::mat train_errors_;
  std::tuple<arma::mat, arma::mat>  stats_;
};

} // namespace classification
} // namespace src

#include "lcurve_hpt_impl_legacy.h"
#endif
