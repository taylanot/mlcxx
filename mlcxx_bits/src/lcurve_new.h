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
          class O=DTYPE >
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
   * @param name        : name the object later on used for saving etc...
   *
   */
  LCurve ( const DATASET& dataset,
           const arma::Row<size_t>& Ns,
           const size_t repeat,
           const bool parallel = false, 
           const bool prog = false,
           const std::filesystem::path path = "./",
           const std::string name = "LCurve" );

  /* Learning Curve Generator
   *
   * @param Ns          : row vector of training points 
   * @param repeat      : amount of times the training for single N takes place
   * @param parallel    : boolean for using parallel computations 
   * @param prog        : boolean for showing the progrress bar
   * @param name        : name the object later on used for saving etc...
   *
   */
  LCurve ( const DATASET& trainset,
           const DATASET& testset,
           const arma::Row<size_t>& Ns,
           const size_t repeat,
           const bool parallel = false, 
           const bool prog = false,
           const std::filesystem::path path = "./",
           const std::string name = "LCurve" );

  /* Generate Learning Curves 
   *
   * @param dataset   : whole large dataset inputs
   * @param args      : possible arguments for the model initialization
   *
   */
  template<template<class,class,class,class,class> class CV = mlpack::SimpleCV,
           class OPT = ens::GridSearch,
           class T = typename std::conditional<
                        std::is_same<CV<MODEL,LOSS,OPT,O,O>,
           mlpack::SimpleCV<MODEL,LOSS,OPT,O,O>>::value,O,size_t>::type,
           class... Ts>
  void Generate ( const T cvp,
                  const Ts&... args );

  /* template<class CV,class OPT = ens::GridSearch, class... Ts> */
  /* void Generate ( const O cvp, */
  /*                 const Ts&... args ); */

  template<class... Ts>
  void Generate ( const Ts&... args );

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
        CEREAL_NVP(testset_),
        CEREAL_NVP(num_class_),
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

  /* Clean Up Method for the over-time processes */
  void _CleanUp( );

  /* Registering the signal handler for some easy stopping. */
  void _RegisterSignalHandler ( );


private: 

  template<template<class,class,class,class,class> class CV,
           class OPT,class Tin=arma::Mat<O>,class Tlab,class T>
  auto _GetHpt ( const Tin& Xtrn, const Tlab& ytrn, const T cvp );

  template<class Tin, class Tlab, class...Ts>
  auto _GetModel ( const Tin& Xtrn, const Tlab& ytrn, const Ts&... args );

  void _CheckStatus( );

  /* Split your dataset
   *
   * @param dataset : whole dataset
   */
  void _SplitData( const DATASET& dataset );

  /* Registering the signal handler for some easy stopping. */
  static void _SignalHandler ( int sig );

  size_t repeat_;
  arma::Row<size_t> Ns_;
  bool parallel_;
  bool prog_;
  std::string name_;
  std::filesystem::path path_;

  std::unordered_map<size_t,std::pair<DATASET,DATASET> > data_;
  arma::uvec jobs_;

  std::optional<DATASET> testset_;
  std::optional<size_t> num_class_;

  LOSS loss_;
  SPLIT split_;
  /* std::optional<std::variant<O,size_t>> cvp_; */
  /* arma::Mat<O> test_errors_; */
};

} // namespace lcurve_



#include "lcurve_impl_new.h"

#endif
