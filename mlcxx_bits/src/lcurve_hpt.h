/**
 * @file lcurve_hpt.h
 * @author Ozgur Taylan Turan
 *
 * Learning curve generator for an MLALgorithm given a dataset 
 *
 */

#ifndef LCURVE_HPT_H
#define LCURVE_HPT_H

namespace src {

//=============================================================================
// LCurve 
//=============================================================================
template<class MODEL,
         class LOSS,
         template<typename, typename, typename, typename, typename> class CV
                                                             = mlpack::SimpleCV,
         class OPT = ens::GridSearch,
         class O=DTYPE>
class LCurveHPT
{
public:

  /* Learning Curve Generator
   *
   * @param Ns          : row vector of training points 
   * @param repeat      : amount of times the training for single N takes place
   * @param cvp         : parameter for cross-validation used
   * @param parallel    : boolean for using parallel computations 
   * @param prog        : boolean for showing the progrress bar
   *
   */
using CVP = typename std::conditional<
        std::is_same<CV<MODEL, LOSS, OPT, O, O>, mlpack::SimpleCV<MODEL, LOSS, OPT, O, O>>::value,
        DTYPE, // Use DTYPE if CV is SimpleCV
        size_t // Otherwise, use size_t
    >::type;

  LCurveHPT( const arma::irowvec& Ns,
             const size_t repeat,
             const CVP cvp=0.2,
             const bool parallel = false, 
             const bool prog = false );

  /* Generate Learning Curves with bootstrapping
   *
   * @param dataset   : whole large dataset 
   * @param args      : possible arguments for the model initialization
   *
   */
         
  template<class T, class... Ts>
  void Bootstrap ( const T& dataset,
                   const Ts&... args );

  /* Generate Learning Curves with random set selection
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
  /* Generate Learning Curves with random set selection
   *
   * @param dataset   : whole large dataset 
   * @param args      : possible arguments for the model initialization
   *
   */
  template<class SPLIT=data::N_Split,class T, class... Ts>
  void RandomSet ( const T& dataset,
                   const Ts&... args );

  /* Generate Learning Curves with random set selection
   *
   * @param inputs    : whole large dataset inputs
   * @param labels    : whole large dataset labels
   * @param args      : possible arguments for the model initialization
   *
   */
  template<class SPLIT=data::N_Split,class T, class... Ts>
  void RandomSet ( const arma::Mat<O>& inputs,
                   const T& labels,
                   const Ts&... args );

  /* Generate Learning Curves with data point addition
   *
   * @param dataset   : whole large dataset 
   * @param args      : possible arguments for the model initialization
   *
   */
  template<class SPLIT=data::N_Split,class T, class... Ts>
  void Additive ( const T& dataset,
                  const Ts&... args );


  /* Generate Learning Curves with data point addition
   *
   * @param inputs    : whole large dataset inputs
   * @param labels    : whole large dataset labels
   * @param args      : possible arguments for the model initialization
   *
   */
  template<class SPLIT=data::N_Split,class T, class... Ts>
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
  template<class SPLIT=data::N_Split,class T, class... Ts>
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
  arma::mat GetResults (  ) {return test_errors_;}
  
private:

  /* 
   * Get the Hyper-parameter tuning object depending on the model
   *
   * @param Xtrn : inputs used for hyper-parmeter tuning
   * @param ytrn : labels used for hyper-parameter tuning

   */
  template<class T>
  mlpack::HyperParameterTuner<MODEL,LOSS,CV,OPT,arma::Mat<O>>
    _GetHpt (const arma::Mat<DTYPE>& Xtrn, const T& ytrn);

  size_t repeat_;
  arma::irowvec Ns_;
  bool parallel_;
  bool save_;
  bool prog_;
  bool save_data_;
  std::string name_;
  arma::Mat<O> results_;
  CVP cvp_;
  arma::Mat<O> test_errors_;

};

} // namespace src


#include "lcurve_hpt_impl.h"

#endif
