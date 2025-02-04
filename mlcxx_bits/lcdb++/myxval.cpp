/**
 * @file myxval.cpp
 * @author Ozgur Taylan Turan
 *
 * Let's check trying to create xval with repitition so that I have 
 * fixed datasets for this i can just use ids to call the dataset portion
 *
 */

#include <headers.h>


using LREG  = algo::classification::LogisticRegression<>;  
using LSVC  = algo::classification::SVM<mlpack::LinearKernel>; 
using GSVC  = algo::classification::SVM<mlpack::GaussianKernel>; 
using ESVC  = algo::classification::SVM<mlpack::EpanechnikovKernel>; 
using LDC   = algo::classification::LDC<>; 
using QDC   = algo::classification::QDC<>; 
using NNC   = algo::classification::NNC<>; 
using OpenML = data::classification::oml::Dataset<>;
using Dataset = data::classification::Dataset<>;
using NMC   = algo::classification::NMC<>; 
using RFOR  = mlpack::RandomForest<>; 
using DT    = mlpack::DecisionTree<>; 
using NB    = mlpack::NaiveBayesClassifier<>;
typedef mlpack::FFN<mlpack::CrossEntropyError,mlpack::HeInitialization> Network;
using ANN = algo::ANN<Network> ;

//-----------------------------------------------------------------------------
//   main : our main function no need to do anything here...
//-----------------------------------------------------------------------------

int main ( int argc, char** argv )
{
  arma::wall_clock timer;

  size_t did = 1046;size_t rep = 1000;
  OpenML dataset(did);
  arma::Mat<DTYPE> X; 
  arma::Row<size_t> y; 

  timer.tic();
  arma::uvec a;
  for(size_t i=0;i<rep;i++)
    a = arma::randi<arma::uvec>(dataset.size_,arma::distr_param(0,dataset.size_));
  PRINT(timer.toc());

  timer.tic();
  for(size_t i=0;i<rep;i++)
    mlpack::ShuffleData(dataset.inputs_, dataset.labels_,X,y);
  PRINT(timer.toc());

  /* xval::KFoldCV<LDC, mlpack::Accuracy> xv(2, dataset.inputs_, dataset.labels_,true); */
  /* PRINT(xv.TrainAndEvaluate(dataset.num_class_,0.)); */
}


