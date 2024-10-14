/**
 * @file config.h
 * @author Ozgur Taylan Turan
 *
 * This file for the configuration of the lcdb++ main file
 *
 */

#ifndef LCDB_CONFIG
#define LCDB_CONFIG

namespace lcdb
{
  // Models 
  using LREG  = algo::classification::LogisticRegression<>;  // x
  using NMC   = algo::classification::NMC<>; 
  using NNC   = algo::classification::NNC<>; 
  using LDC   = algo::classification::LDC<>; 
  using QDC   = algo::classification::QDC<>; 
  using LSVC  = algo::classification::SVM<mlpack::LinearKernel>; 
  using GSVC  = algo::classification::SVM<mlpack::GaussianKernel>; 
  using ESVC  = algo::classification::SVM<mlpack::EpanechnikovKernel>; 
  using ADAB  = mlpack::AdaBoost<>; 
  using RFOR  = mlpack::RandomForest<>; 
  using DT    = mlpack::DecisionTree<>; 
  using NB    = mlpack::NaiveBayesClassifier<>;

  // Losses
  using Acc = mlpack::Accuracy;
  using Crs = utils::CrossEntropy;
  using Auc = utils::AUC;
  using Bri = utils::BrierLoss;

  // validation size for the Hyper-parameter optimization
  DTYPE vsize = 0.2;

  // lambdas: for the QDC, LDC, LREG
  auto lambdas = arma::logspace<arma::Row<DTYPE>>(-8,1,3);
  // Cs: GSVC,LSVC,ESVC
  auto Cs = arma::logspace<arma::Row<DTYPE>>(-2,1,2);
  // ns: NNC
  auto ns = arma::regspace<arma::Row<DTYPE>>(1,1,10);
  // maxiters : ADAB
  auto maxiters = arma::regspace<arma::Row<DTYPE>>(1,10,100);
  // numtrees : RFOR
  auto numtrees = arma::regspace<arma::Row<DTYPE>>(1,10,100);
  // leafsizes: DT
  auto numtrees = arma::regspace<arma::Row<DTYPE>>(1,10,100);


}


#endif 
