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
         class LOSS,
         class O=DTYPE>
class LCurve
{
public:

  /* Learning Curve Generator
   *
   * @param Ns          : row vector of training points 
   * @param repeat      : amount of times the training for single N takes place
   * @param parallel    : boolean for using parallel computations 
   * @param prog        : boolean for showing the progrress bar
   *
   */
  LCurve ( const arma::irowvec& Ns,
           const double repeat,
           const bool parallel = false, 
           const bool prog = false );

  /* Generate Learning Curves with RandomSet Selection
   *
   * @param dataset   : whole large dataset inputs
   * @param args      : possible arguments for the model initialization
   *
   */
  template<class T, class... Ts>
  void Bootstrap ( const T& dataset,
                   const Ts&... args );

  /* Generate Learning Curves with RandomSet Selection
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


  /* Generate Learning Curves with RandomSet Selection
   *
   * @param inputs    : whole large dataset inputs
   * @param labels    : whole large dataset labels
   * @param args      : possible arguments for the model initialization
   *
   */
  template<class SPLIT=data::N_Split, class T, class... Ts>
  void RandomSet ( const arma::Mat<O>& inputs,
                   const T& labels,
                   const Ts&... args );

  /* Generate Learning Curves with RandomSet Selection
   *
   * @param dataset   : whole large dataset
   * @param args      : possible arguments for the model initialization
   *
   */
  template<class SPLIT=data::N_Split, class T, class... Ts>
  void RandomSet ( const T& dataset,
                   const Ts&... args );

  /* Generate Learning Curves with data point addition
   *
   * @param inputs    : whole large dataset inputs
   * @param labels    : whole large dataset labels
   * @param args      : possible arguments for the model initialization
   *
   */
  template<class SPLIT=data::N_Split, class T, class... Ts>
  void Additive ( const T& dataset,
                  const Ts&... args );

  /* Generate Learning Curves with data point addition
   *
   * @param dataset   : whole large dataset 
   * @param args      : possible arguments for the model initialization
   *
   */
  template<class SPLIT=data::N_Split, class T, class... Ts>
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
  template<class SPLIT=data::N_Split, class T, class... Ts>
  void Split ( const T& trainset,
               const T& testset,
               const Ts&... args );

  /* Get the results of the learning curve generation
   *
   */

  arma::Mat<O> GetResults (  ) const  {return test_errors_;}

private:
  /* SPLIT split_; */
  size_t repeat_;
  arma::irowvec Ns_;
  bool parallel_;
  bool prog_;
  arma::Mat<O> results_;

  LOSS loss_;

  arma::Mat<O> test_errors_;

};

} // namespace src


#include "lcurve_impl.h"

#endif

