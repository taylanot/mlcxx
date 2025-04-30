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
  // Datasets
  using DATASET = data::oml::Dataset<size_t>;
  using FIDEL = float;

  // Models 
  using NMC   = algo::classification::NMC<>; 
  using NB    = mlpack::NaiveBayesClassifier<>;

  using LREG  = algo::classification::LogisticRegression<>;
  using NNC   = algo::classification::NNC<>; 
  using LDC   = algo::classification::LDC<>; 
  using QDC   = algo::classification::QDC<>; 
  using LSVC  = mlpack::LinearSVM<>; 
  using GSVC  = algo::classification::SVM<mlpack::GaussianKernel,0>; 
  using ADAB  = mlpack::AdaBoost<mlpack::ID3DecisionStump>; 
  using RFOR  = mlpack::RandomForest<>; 
  using DT    = mlpack::DecisionTree<>; 

  // Losses
  using ACC = mlpack::Accuracy;
  using CRS = utils::CrossEntropy;
  using AUC = utils::AUC;
  using BRI = utils::BrierLoss;
  
  // Samplers
  using RND = data::RandomSelect<arma::uword>;
  using BOO = data::Bootstrap<arma::uword>;
  using ADD = data::Additive<arma::uword>;

  using MODS = std::variant< std::type_identity<NMC>,
                             std::type_identity<NNC>,
                             std::type_identity<LREG>,
                             std::type_identity<LDC>,
                             std::type_identity<QDC>,
                             std::type_identity<LSVC>,
                             std::type_identity<GSVC>,
                             std::type_identity<ADAB>,
                             std::type_identity<RFOR>,
                             std::type_identity<DT> >;

  using SMPS = std::variant< std::type_identity<BOO>,
                             std::type_identity<RND>,
                             std::type_identity<ADD> >;

  using LOSS = std::variant< std::type_identity<ACC>,
                             std::type_identity<AUC>,
                             std::type_identity<BRI>,
                             std::type_identity<CRS>>;

  // Dispatcher maps from string to type_identity
  const std::unordered_map<std::string, MODS> models = 
  { {"nmc", std::type_identity<NMC>{}},
    {"nnc", std::type_identity<NNC>{}},
    {"lreg", std::type_identity<LREG>{}},
    {"ldc", std::type_identity<LDC>{}},
    {"qdc", std::type_identity<QDC>{}},
    {"lsvc", std::type_identity<LSVC>{}},
    {"gsvc", std::type_identity<GSVC>{}},
    {"adab", std::type_identity<ADAB>{}},
    {"dt", std::type_identity<DT>{}},
    {"rfor", std::type_identity<RFOR>{}},
  };

  const std::unordered_map<std::string, SMPS> samples = 
  { {"rands", std::type_identity<RND>{}},
    {"boot", std::type_identity<BOO>{}},
    {"add", std::type_identity<ADD>{}} };

  const std::unordered_map<std::string, LOSS> losses = 
  { {"acc", std::type_identity<ACC>{}},
    {"auc", std::type_identity<AUC>{}},
    {"bri", std::type_identity<BRI>{}},
    {"crs", std::type_identity<CRS>{}} };



  // validation size for the Hyper-parameter optimization
  static const DTYPE vsize = 0.2;
  // train-test split size if you are not using bootstrap or randomset
  static const DTYPE splitsize = 0.2;

  using HptSet = std::variant<arma::Row<DTYPE>, arma::Row<size_t>>;
  HptSet get_hptset(const std::string &model_name)
  {
    const std::unordered_set<std::string> logspace_models = {
      "nmc", "ldc", "qdc", "lreg", "lsvc", "gsvc"
    };

    if (logspace_models.contains(model_name))
      return arma::logspace<arma::Row<DTYPE>>(-8, 2, 11);
    else
      return arma::regspace<arma::Row<size_t>>(1, 1, 10);
  };
  // hpt option for the QDC, LDC, LREG and Cs GSVC,LSVC,ESVC
  static const auto hptset1 = arma::logspace<arma::Row<DTYPE>>(-8,2,11);

  // hpt option for NNC, AdaBoost, RandomForest, DecisionTree
  static auto hptset2 = arma::regspace<arma::Row<DTYPE>>(1,1,10);

  // Where to save the experiments
  /* static const std::filesystem::path path = EXP_PATH/"lcdb++"; */
  static const std::filesystem::path path = "EXPERIMENTS/lcdb++";

}

/* using DATASET = data::oml::Dataset<>; */
/* namespace lcdb */
/* { */
/*   // Dataset */
/*   // Models */ 
/*   using LREG  = algo::classification::LogisticRegression<>;  // x */
/*   using NMC   = algo::classification::NMC<>; */ 
/*   using NNC   = algo::classification::NNC<>; */ 
/*   using LDC   = algo::classification::LDC<>; */ 
/*   using QDC   = algo::classification::QDC<>; */ 
/*   using LSVC  = algo::classification::SVM<mlpack::LinearKernel,0>; */ 
/*   using GSVC  = algo::classification::SVM<mlpack::GaussianKernel,0>; */ 
/*   using ADAB  = mlpack::AdaBoost<mlpack::ID3DecisionStump>; */ 
/*   using RFOR  = mlpack::RandomForest<>; */ 
/*   using DT    = mlpack::DecisionTree<>; */ 
/*   using NB    = mlpack::NaiveBayesClassifier<>; */

/*   // Losses */
/*   using Acc = mlpack::Accuracy; */
/*   using Crs = utils::CrossEntropy; */
/*   using Auc = utils::AUC; */
/*   using Bri = utils::BrierLoss; */

/*   // validation size for the Hyper-parameter optimization */
/*   static const DTYPE vsize = 0.2; */
/*   // train-test split size if you are not using bootstrap or randomset */
/*   static const DTYPE splitsize = 0.2; */

/*   // lambdas: for the QDC, LDC, LREG and Cs GSVC,LSVC,ESVC */
/*   static const auto lambdas = arma::logspace<arma::Row<DTYPE>>(-8,2,11); */
/*   static const auto Cs = arma::logspace<arma::Row<DTYPE>>(-8,2,11); */

/*   /1* auto Cs = arma::logspace<arma::Row<DTYPE>>(-8,1,3); *1/ */

/*   /1* // ns: NNC *1/ */
/*   /1* auto ns = arma::regspace<arma::Row<DTYPE>>(1,1,10); *1/ */
/*   /1* // maxiters : ADAB *1/ */
/*   /1* auto maxiters = arma::regspace<arma::Row<DTYPE>>(1,10,100); *1/ */
/*   /1* // numtrees : RFOR *1/ */
/*   /1* auto numtrees = arma::regspace<arma::Row<DTYPE>>(1,10,100); *1/ */
/*   /1* // leafsizes: DT *1/ */
/*   /1* auto leafs = arma::regspace<arma::Row<DTYPE>>(1,10,100); *1/ */

/*   // Where to save the experiments */
/*   /1* static const std::filesystem::path path = EXP_PATH/"lcdb++"; *1/ */
/*   static const std::filesystem::path path = "lcdb++_svcs"; */
/*   // Number of instance limit */ 
/*   static const size_t nlim = 2000; */
/*   // Number of features limit */ 
/*   static const size_t flim = 50; */
/*   // Ns that we want to investigate */
/*   static const arma::irowvec Ns = arma::regspace<arma::irowvec> */
/*                                               (1,1,100); */
/*   static const arma::irowvec hptNs = arma::regspace<arma::irowvec> */
/*                                               (10,1,100); */

/* } */


#endif 
