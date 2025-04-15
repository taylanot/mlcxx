/**
 * @file lcurve.h
 * @author Ozgur Taylan Turan
 *
 * Learning curve generator for an MLALgorithm given a dataset 
 *
 */

#ifndef LCURVE_NEW_H
#define LCURVE_NEW_H

namespace lcurve {

// Global function pointer this is for cleanup-related things.
std::function<void()> _globalSafeFailFunc;  
//=============================================================================
// LCurve 
//=============================================================================
template< class MODEL,
          class DATASET,
          class SPLIT,
          class LOSS,
          class O=DTYPE>
class LCurve
{
public:

  /* Learning Curve Generator Empty Constructor */
  LCurve ( ) { };

  /* Learning Curve Generator
   *
   * @param Ns          : row vector of training points 
   * @param repeat      : amount of times the training for single N takes place
   * @param parallel    : boolean for using parallel computations 
   * @param prog        : boolean for showing the progrress bar
   *
   */
  LCurve ( const arma::Row<size_t>& Ns,
           const size_t repeat,
           const bool parallel = false, 
           const bool prog = false,
           const std::string name = "LCurve" );

  /* Generate Learning Curves 
   *
   * @param dataset   : whole large dataset inputs
   * @param args      : possible arguments for the model initialization
   *
   */
  template<class... Ts>
  void Generate ( const DATASET& dataset,
                  const Ts&... args );

  /* Generate Learning Curves with RandomSet Selection
   *
   * @param inputs    : whole large dataset inputs
   * @param labels    : whole large dataset labels
   * @param args      : possible arguments for the model initialization
   *
   */
  template<class T, class... Ts>
  void Generate ( const arma::Mat<O>& inputs,
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
  void Generate ( const T& trainset,
                  const T& testset,
                  const Ts&... args );

  /* Continue Learning Curve Generation
   *
   *
   */
  template<class... Ts>
  void Continue ( const Ts&... args );

public: 
  /* Get the results of the learning curve generation */
  arma::Mat<O> GetResults (  ) const  {return test_errors_;}

  /* Serliazation with cereal for the class. */
  template <class Archive>
  void serialize(Archive& ar) 
  {
    ar( CEREAL_NVP(repeat_),
        CEREAL_NVP(Ns_),
        CEREAL_NVP(parallel_),
        CEREAL_NVP(prog_),
        CEREAL_NVP(name_),
        CEREAL_NVP(data_),
        CEREAL_NVP(jobs_),
        CEREAL_NVP(test_errors_));
  }

  /* Save the object to a BinaryFile
   *
   * @param filename : binary file name
   */
  void Save ( const std::string& filename );

  /* Load the object from a BinaryFile
   *
   * @param filename : binary file name
   */
  static std::shared_ptr<LCurve<MODEL,DATASET,SPLIT,LOSS,O>> Load 
                                              ( const std::string& filename ); 
  arma::Mat<O> test_errors_;
private: 
  void _CheckStatus( );
  /* Split your data */
  void _SplitData( DATASET dataset );

  /* Clean Up Method for the over-time processes */
  void _CleanUp( );

  /* Registering the signal handler for some easy stopping. */
  void _RegisterSignalHandler ( );

  /* Registering the signal handler for some easy stopping. */
  static void _SignalHandler ( int sig );

  size_t repeat_;
  arma::Row<size_t> Ns_;
  bool parallel_;
  bool prog_;
  std::string name_;
  std::unordered_map<size_t,std::pair<DATASET,DATASET> > data_;
  arma::uvec jobs_;

  LOSS loss_;
  SPLIT split_;

  /* arma::Mat<O> test_errors_; */
};

} // namespace src


#include "lcurve_impl_new.h"

#endif
