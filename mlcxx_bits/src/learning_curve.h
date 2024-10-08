/**
 * @file learning_curve.h
 * @author Ozgur Taylan Turan
 *
 * Learning curve generator for an MLALgorithm given a dataset 
 *
 * TODO: 
 * - Should we go for an inherited version? Will cost compilation time though?
 * - Write it much cleaner.
 *
 *
 */

#ifndef LEARNING_CURVE_H
#define LEARNING_CURVE_H

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

//=============================================================================
// LCurve 
//=============================================================================

template<class MODEL,
         class LOSS>
class LCurve
{
public:

  /* Learning Curve Generator
   *
   * @param Ns          : row vector of training points 
   * @param repeat      : amount of times the training for single N takes place
   *
   */
  LCurve<MODEL,LOSS> ( const arma::irowvec& Ns,
                       const double& repeat );

  /* Generate Learning Curve and save all the data
   *
   * @param save_all  : save all the information regarding the generation
   * @param filename  : filename with extension to save
   * @param inputs    : whole large dataset inputs
   * @param labels    : whole large dataset labels
   * @param args      : possible arguments for the model initialization
   *
   */
  template<class... Ts>
  void Generate ( const bool save_all,
                  const std::filesystem::path filename,
                  const arma::mat& inputs,
                  const arma::rowvec& labels,
                  const Ts&... args );

  template<class... Ts>
  void Generate ( const bool save_all,
                  const std::filesystem::path filename,
                  const arma::mat& inputs,
                  const arma::mat& labels,
                  const Ts&... args );

  /* Generate Learning Curve and save statistics to a file
   *
   * @param filename  : filename with extension to save
   * @param inputs    : whole large dataset inputs
   * @param labels    : whole large dataset labels
   * @param args      : possible arguments for the model initialization
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

  /* Generate Learning Curves 
   *
   * @param trainset  : inputs and labels for training
   * @param testset   : inputs and labels for testing
   * @param args      : possible arguments for the model initialization
   *
   */
  template<class... Ts>
  void Generate ( const utils::data::regression::Dataset& trainset,
                  const utils::data::regression::Dataset& testset,
                  const Ts&... args );
  template<class... Ts>
  void ParallelGenerate ( const utils::data::regression::Dataset& trainset,
                          const utils::data::regression::Dataset& testset,
                          const Ts&... args );

  /* Save the stats to a file
   *
   * @param filename  : filename with extension to save
   *
   */
  template<class... Ts>
  void Save ( const std::filesystem::path filename );

  /* Get The Results
   *
   */
  arma::mat GetResults (  ) {return results_;}

private:
  size_t repeat_;
  arma::irowvec Ns_;
  arma::mat results_;

public:
  arma::mat test_errors_;
  arma::mat train_errors_;
  std::tuple<arma::mat, arma::mat>  stats_;

};

//=============================================================================
// VariableLCurve
//=============================================================================

template<class MODEL,
         class LOSS>
class VariableLCurve
{
public:

  /* Learning Curve Generator
   *
   * @param Ns          : row vector of training points 
   * @param repeat      : amount of times the training for single N takes place
   *
   */
  VariableLCurve<MODEL,LOSS> ( const arma::irowvec& Ns,
                               const arma::irowvec& repeat );

  /* Generate Learning Curve and save all the data
   *
   * @param save_all  : save all the information regarding the generation
   * @param filename  : filename with extension to save
   * @param inputs    : whole large dataset inputs
   * @param labels    : whole large dataset labels
   * @param args      : possible arguments for the model initialization
   *
   */
  template<class... Ts>
  void Generate ( const bool save_all,
                  const std::filesystem::path filename,
                  const arma::mat& inputs,
                  const arma::rowvec& labels,
                  const Ts&... args );

  template<class... Ts>
  void Generate ( const bool save_all,
                  const std::filesystem::path filename,
                  const arma::mat& inputs,
                  const arma::mat& labels,
                  const Ts&... args );

  /* Generate Learning Curve and save statistics to a file
   *
   * @param filename  : filename with extension to save
   * @param inputs    : whole large dataset inputs
   * @param labels    : whole large dataset labels
   * @param args      : possible arguments for the model initialization
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
  void Save ( const std::filesystem::path filename );

  /* Get The Results
   *
   */
  arma::mat GetResults (  ) {return results_;}

private:
  arma::irowvec repeat_;
  arma::irowvec Ns_;
  arma::mat results_;

public:
  //std::map<size_t, arma::mat> test_errors_;
  //std::map<size_t, arma::mat> train_errors_;
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

//=============================================================================
// LCurve 
//=============================================================================

template<class MODEL,
         class LOSS>
class LCurve
{
public:

  /* Learning Curve Generator
   *
   * @param Ns          : row vector of training points 
   * @param repeat      : amount of times the training for single N takes place
   *
   */
  LCurve<MODEL,LOSS> ( const arma::irowvec& Ns,
                       const double& repeat );

  /* Generate Learning Curve and save statistics to a file
   *
   * @param filename  : filename with extension to save
   * @param inputs    : whole large dataset inputs
   * @param labels    : whole large dataset labels
   * @param args      : possible arguments for the model initialization
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

  /* Generate Learning Curves 
   *
   * @param trainset  : inputs and labels for training
   * @param testset   : inputs and labels for testing
   * @param args      : possible arguments for the model initialization
   *
   */
  template<class... Ts>
  void Generate ( const utils::data::classification::Dataset& trainset,
                  const utils::data::classification::Dataset& testset,
                  const Ts&... args );

  
  /* Save the stats to a file
   *
   * @param filename  : filename with extension to save
   *
   */
  template<class... Ts>
  void Save ( const std::filesystem::path filename );

  /* Get The Results
   *
   */
  arma::mat GetResults (  ) {return results_;}



private:
  size_t repeat_;
  arma::irowvec Ns_;
  arma::mat results_;

public:
  arma::mat test_errors_;
  arma::mat train_errors_;
  std::tuple<arma::mat, arma::mat>  stats_;

};

} // namespace classification
} // namespace src


#include "learning_curve_impl.h"

#endif

