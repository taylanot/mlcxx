/**
 * @file config.h
 * @author Ozgur Taylan Turan
 *
 * This file for the configuration of the lcdb++ main file
 *
 */

#ifndef LCDB_CONFIG
#define LCDB_CONFIG

using DATASET = data::oml::Dataset<>;
namespace lcdb
{
  // Dataset
  // Models 
  using LREG  = algo::classification::LogisticRegression<>;  // x
  using NMC   = algo::classification::NMC<>; 
  using NNC   = algo::classification::NNC<>; 
  using LDC   = algo::classification::LDC<>; 
  using QDC   = algo::classification::QDC<>; 
  using LSVC  = algo::classification::SVM<mlpack::LinearKernel,0>; 
  using GSVC  = algo::classification::SVM<mlpack::GaussianKernel,0>; 
  using ADAB  = mlpack::AdaBoost<mlpack::ID3DecisionStump>; 
  using RFOR  = mlpack::RandomForest<>; 
  using DT    = mlpack::DecisionTree<>; 
  using NB    = mlpack::NaiveBayesClassifier<>;

  // Losses
  using Acc = mlpack::Accuracy;
  using Crs = utils::CrossEntropy;
  using Auc = utils::AUC;
  using Bri = utils::BrierLoss;

  // validation size for the Hyper-parameter optimization
  static const DTYPE vsize = 0.2;
  // train-test split size if you are not using bootstrap or randomset
  static const DTYPE splitsize = 0.2;

  // lambdas: for the QDC, LDC, LREG and Cs GSVC,LSVC,ESVC
  static const auto lambdas = arma::logspace<arma::Row<DTYPE>>(-8,2,11);
  static const auto Cs = arma::logspace<arma::Row<DTYPE>>(-8,2,11);

  /* auto Cs = arma::logspace<arma::Row<DTYPE>>(-8,1,3); */

  /* // ns: NNC */
  /* auto ns = arma::regspace<arma::Row<DTYPE>>(1,1,10); */
  /* // maxiters : ADAB */
  /* auto maxiters = arma::regspace<arma::Row<DTYPE>>(1,10,100); */
  /* // numtrees : RFOR */
  /* auto numtrees = arma::regspace<arma::Row<DTYPE>>(1,10,100); */
  /* // leafsizes: DT */
  /* auto leafs = arma::regspace<arma::Row<DTYPE>>(1,10,100); */

  // Where to save the experiments
  /* static const std::filesystem::path path = EXP_PATH/"lcdb++"; */
  static const std::filesystem::path path = "lcdb++_svcs";
  // Number of instance limit 
  static const size_t nlim = 2000;
  // Number of features limit 
  static const size_t flim = 50;
  // Ns that we want to investigate
  static const arma::irowvec Ns = arma::regspace<arma::irowvec>
                                              (1,1,100);
  static const arma::irowvec hptNs = arma::regspace<arma::irowvec>
                                              (10,1,100);

}


#endif 
