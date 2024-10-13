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
  using LREG  = algo::classification::LogisticRegression<>;
  using NMC   = algo::classification::NMC<>;
  using LDC   = algo::classification::LDC<>;
  using QDC   = algo::classification::QDC<>;
  using NB    = mlpack::NaiveBayesClassifier<>;
  using LSVC  = algo::classification::SVM<mlpack::LinearKernel>;
  using GSVC  = algo::classification::SVM<mlpack::GaussianKernel>;
  using ESVC  = algo::classification::SVM<mlpack::EpanechnikovKernel>;
  using ADAB  = mlpack::AdaBoost<>;
  using RFOR  = mlpack::RandomForest<>;
  using DecisionTree = mlpack::DecisionTree<>;

  // Losses
  using Acc = mlpack::Accuracy;
  using Crs = utils::CrossEntropy;
  using Auc = utils::AUC;
  using Auc = utils::BrierLoss;

}


#endif 
