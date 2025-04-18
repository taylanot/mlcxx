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
          template<class,class,class,class,class> class CV= mlpack::SimpleCV,
          class OPT=ens::GridSearch,
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
   * @param name        : name the object later on used for saving etc...
   *
   */
  LCurve ( const arma::Row<size_t>& Ns,
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
  using CVP = typename std::conditional<
        std::is_same<CV<MODEL, LOSS, OPT, O, O>,
                     mlpack::SimpleCV<MODEL, LOSS, OPT, O, O>>::value,
        DTYPE, // Use DTYPE if CV is SimpleCV
        size_t // Otherwise, use size_t
    >::type;
  LCurve ( const arma::Row<size_t>& Ns,
           const size_t repeat,
           const CVP cvp = 0.2,
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
  template<class... Ts>
  void Generate ( const DATASET& dataset,
                  const Ts&... args );

  /* Generate Learning Curves with Train and Test Splitted Datasets
   *
   * @param trainset  : dataset used for training 
   * @param testset   : dataset used for testing 
   * @param args      : possible arguments for the model initialization
   *
   */
  template<class... Ts>
  void Generate ( const DATASET& trainset,
                  const DATASET& testset,
                  const Ts&... args );


  /* Continue Learning Curve Generation
   *
   * @param args      : possible arguments for the model initialization
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
  static std::shared_ptr<LCurve<MODEL,DATASET,SPLIT,LOSS,CV,OPT,O>> Load 
                                              ( const std::string& filename ); 
  arma::Mat<O> test_errors_;

private: 

  template<class T>
  mlpack::HyperParameterTuner<MODEL,LOSS,CV,OPT,arma::Mat<O>>
    _GetHpt (const arma::Mat<DTYPE>& Xtrn, const T& ytrn);

  void _CheckStatus( );

  /* Split your dataset
   *
   * @param dataset : whole dataset
   */
  void _SplitData( const DATASET& dataset );

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
  std::filesystem::path path_;

  std::unordered_map<size_t,std::pair<DATASET,DATASET> > data_;
  arma::uvec jobs_;
  std::optional<DATASET> testset_;

  LOSS loss_;
  SPLIT split_;
  std::optional<CVP> cvp_;

  /* arma::Mat<O> test_errors_; */
};

} // namespace src


#include "lcurve_impl_new.h"

#endif
