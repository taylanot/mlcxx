/**
 * @file lcurve.h
 * @author Ozgur Taylan Turan
 *
 * Learning curve generator for an MLALgorithm given a dataset 
 *
 */

#ifndef LCURVE_H
#define LCURVE_H

namespace src {

//=============================================================================
// LCurve 
//=============================================================================
template<class MODEL,
         class LOSS,class SPLIT=utils::Split, class O=DTYPE>
class LCurve
{
public:

  /* Learning Curve Generator
   *
   * @param Ns          : row vector of training points 
   * @param repeat      : amount of times the training for single N takes place
   * @param parallel    : boolean for using parallel computations 
   * @param save        : boolean for saving the results
   * @param prog        : boolean for showing the progrress bar
   * @param save_data   : boolean for saving the data used for train and test
   * @param name        : name of the experiment
   *
   */
  LCurve( const arma::irowvec& Ns,
          const double repeat,
          const bool parallel = false, 
          const bool save = false,
          const bool prog = false,
          const std::string name = "lcurve",
          const bool save_data = false );

  /* Generate Learning Curves with Bootstrap
   *
   * @param inputs    : whole large dataset inputs
   * @param labels    : whole large dataset labels
   * @param args      : possible arguments for the model initialization
   *
   */
  template<class T, class... Ts>
  void Bootstrap ( const arma::Mat<O>& inputs,
                   const T& labels,
                   const Ts&... args );

  /* Generate Learning Curves with data point addition
   *
   * @param inputs    : whole large dataset inputs
   * @param labels    : whole large dataset labels
   * @param args      : possible arguments for the model initialization
   *
   */
  template<class T, class... Ts>
  void Additive ( const arma::Mat<O>& inputs,
                  const T& labels,
                  const Ts&... args );

  /* Generate Learning Curves with Train and Test Splitted Datasets
   *
   * @param trainset  : dataset used for training 
   * @param testset   : dataset used for testing 
   * @param args      : possible arguments for the model initialization
   *
   */
  template<class T, class... Ts>
  void Split ( const T& trainset,
               const T& testset,
               const Ts&... args );

  /* Get the results of the learning curve generation
   *
   * @param trainset  : dataset used for training 
   * @param testset   : dataset used for testing 
   * @param args      : possible arguments for the model initialization
   *
   */

  arma::mat GetResults (  ) {return results_;}

private:
  SPLIT split_;
  size_t repeat_;
  arma::irowvec Ns_;
  bool parallel_;
  bool save_;
  bool prog_;
  bool save_data_;
  std::string name_;
  arma::Mat<O> results_;

public:
  arma::Mat<O> test_errors_;
  arma::Mat<O> train_errors_;
  std::tuple<arma::Mat<O>, arma::Mat<O>>  stats_;

};

} // namespace src


#include "lcurve_impl.h"

#endif

