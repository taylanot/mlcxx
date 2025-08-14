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

//-----------------------------------------------------------------------------
// LCurve 
//-----------------------------------------------------------------------------
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
   * @param dataset     : this is the training and the testset combined
   * @param Ns          : row vector of training points 
   * @param repeat      : amount of times the training for single N takes place
   * @param parallel    : boolean for using parallel computations 
   * @param prog        : boolean for showing the progrress bar
   * @param path        : path for the saving etc...
   * @param name        : name the object later on used for saving etc...
   * @param seed        : for reproduction it is essential
   */
  LCurve ( const DATASET& dataset,
           const arma::Row<size_t>& Ns,
           const size_t repeat,
           const bool parallel = false, 
           const bool prog = false,
           const std::filesystem::path path = "./",
           const std::string name = "LCurve",
           const size_t seed = SEED );

  /* Learning Curve Generator
   *
   * @param trainset    : this is the training set used for sampling
   * @param testset     : seperate testset only used for testing
   * @param Ns          : row vector of training points 
   * @param repeat      : amount of times the training for single N takes place
   * @param parallel    : boolean for using parallel computations 
   * @param prog        : boolean for showing the progrress bar
   * @param path        : path for the saving etc...
   * @param name        : name the object later on used for saving etc...
   * @param seed        : for reproduction it is essential
   */
  LCurve ( const DATASET& trainset,
           const DATASET& testset,
           const arma::Row<size_t>& Ns,
           const size_t repeat,
           const bool parallel = false, 
           const bool prog = false,
           const std::filesystem::path path = "./",
           const std::string name = "LCurve",
           const size_t seed = SEED );

  /* Generate Learning Curves 
   *
   * @param cvp   : this is either number of folds or percent of split
   * @param args  : possible arguments for hpt. optimization of the model
   *
   */
  template<template<class,class,class,class,class> class CV = mlpack::SimpleCV,
           class OPT = ens::GridSearch,
           class T = typename std::conditional<
                        std::is_same<CV<MODEL,LOSS,OPT,O,O>,
           mlpack::SimpleCV<MODEL,LOSS,OPT,O,O>>::value,O,size_t>::type,
           class... Ts>
  void GenerateHpt ( const T cvp,
                     const Ts&... args );

  /* Generate Learning Curves 
   *
   * @param args  : possible arguments for model initialization
   *
   */
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
        CEREAL_NVP(seed_),
        CEREAL_NVP(path_),
        CEREAL_NVP(testset_),
        CEREAL_NVP(trainset_),
        CEREAL_NVP(num_class_),
        CEREAL_NVP(seeds_),
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

  /* Clean Up Method for the over-time processes 
   * __This should not be used outside, it is public just because I want to 
   *    load with safe stopping. */
  void _CleanUp( );

  /* Registering the signal handler for some easy stopping. 
   * __This should not be used outside, it is public just because I want to 
   *    load with safe stopping. */
  void _RegisterSignalHandler ( );

  /* Check the Status of the l.curve generation how many do you have left? */
  bool CheckStatus( bool print = false );

  std::string GetName( );

private: 

  template<template<class,class,class,class,class> class CV,
           class OPT,class Tin=arma::Mat<O>,class Tlab,class T>
  auto _GetHpt ( const Tin& Xtrn, const Tlab& ytrn, const T cvp );

  template<class Tin, class Tlab, class...Ts>
  auto _GetModel ( const Tin& Xtrn, const Tlab& ytrn, const Ts&... args );


  /* Split your dataset and return both the train and test sets.
   *
   * @param dataset : whole dataset
   */
  std::vector<std::pair<arma::uvec,arma::uvec>>
  _SplitData( const DATASET& dataset, const size_t seed );

  /* Registering the signal handler for some easy stopping. */
  static void _SignalHandler ( int sig );

  size_t repeat_; // Times that we repeat the l.curve generation.
  arma::Row<size_t> Ns_; // All the sample sizes used for l.curve generation.
  bool parallel_; // Do you want to have paralelization or not? I say go for it!
  bool prog_; // Triggaring the progress bar. Not essential...
  std::string name_; // Name of the object, this is your escape name 
  std::filesystem::path path_; // Where do you run the experiment from anything,
                               // happens this is your place to look for.
  size_t seed_; // For reproduction purposes...

  // Testset is sometimes there and sometimes not...
  std::optional<DATASET> testset_;
  // This is the set used for training.
  DATASET trainset_;

  // If your learner takes in number of clasesses this is usefull not present
  // for some binary classifiers in mlpack and regression models.
  std::optional<size_t> num_class_;

  // This for initializing the seeds used for every repitition
  // depends on the initial seed 
  arma::irowvec seeds_;

  //  Loss and split objects used for learning curve creation
  LOSS loss_;
  SPLIT split_;
};

} // namespace lcurve

#include "lcurve_impl.h"

#endif
