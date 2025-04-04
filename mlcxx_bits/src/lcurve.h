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

// Global function pointer this is for cleanup-related things.
std::function<void()> globalSafeFailFunc;  
//=============================================================================
// LCurve 
//=============================================================================
template<class MODEL,
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
  LCurve ( const arma::irowvec& Ns,
           const double repeat,
           const bool parallel = false, 
           const bool prog = false,
           const std::string type = "boot",
           const std::string name = "LCurve" );

  /* Generate Learning Curves 
   *
   * @param dataset   : whole large dataset inputs
   * @param args      : possible arguments for the model initialization
   *
   */
  template<class SPLIT=data::N_Split, class T, class... Ts>
  void Generate ( const T& dataset,
                  const Ts&... args );

  /* Generate Learning Curves with RandomSet Selection
   *
   * @param inputs    : whole large dataset inputs
   * @param labels    : whole large dataset labels
   * @param args      : possible arguments for the model initialization
   *
   */
  template<class SPLIT=data::N_Split, class T, class... Ts>
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
  template<class SPLIT=data::N_Split, class T, class... Ts>
  void Generate ( const T& trainset,
                  const T& testset,
                  const Ts&... args );

public:
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
        CEREAL_NVP(loss_),
        CEREAL_NVP(name_),
        CEREAL_NVP(type_),
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
  static std::shared_ptr<LCurve<MODEL,LOSS,O>> Load 
                                              ( const std::string& filename ); 
private: 
  /* Clean Up Method for the over-time processes */
  void CleanUp( );

  /* Registering the signal handler for some easy stopping. */
  void RegisterSignalHandler ( );

  /* Registering the signal handler for some easy stopping. */
  static void SignalHandler ( int sig );

  size_t repeat_;
  arma::irowvec Ns_;
  bool parallel_;
  bool prog_;
  std::string type_;
  std::string name_;

  LOSS loss_;

  arma::Mat<O> test_errors_;
  const std::unordered_map<std::string, int> 
    map_ = { {"boot", 1},
             {"split", 2},
             {"rands", 3} };
};

} // namespace src


#include "lcurve_impl.h"

#endif
