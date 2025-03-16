/**
 * @file xval2.cpp
 * @author Ozgur Taylan Turan
 *
 * Let's check xval results
 */

#include <headers.h>

using MyData = data::regression::Dataset<>;
using OpenML = data::oml::Dataset<size_t>;

using LREG = algo::classification::LogisticRegression<>;
using LDA = algo::classification::LDC<>; 
using QDA = algo::classification::QDC<>;
using LSVM = mlpack::LinearSVM<>;
using ADAB  = mlpack::AdaBoost<mlpack::ID3DecisionStump>; 
using RFOR  = mlpack::RandomForest<>; 
using DT    = mlpack::DecisionTree<>; 
using LSVC  = algo::classification::SVM<mlpack::LinearKernel>; 
using GSVC  = algo::classification::SVM<mlpack::GaussianKernel>; 
using NMC   = algo::classification::NMC<>; 
using NNC   = algo::classification::NNC<>; 


using METRIC = mlpack::Accuracy;
using srowvec = arma::Row<size_t>;
using rowvec = arma::Row<DTYPE>;

template<class MODEL,class DATA=OpenML,class HPTSPACE=arma::Row<DTYPE>>
void RunExperiment ( DATA trainset,
                     DATA testset,
                     size_t fold, size_t dense, size_t reps,
                     HPTSPACE ls )
{
  arma::Row<DTYPE> mean_sel(reps);
  arma::Row<DTYPE> med_sel(reps);
  arma::Row<DTYPE> qlow_sel(reps);
  arma::Row<DTYPE> qhigh_sel(reps);

  #pragma omp parallel for
  for (size_t rep=0; rep < reps; rep++)
  {
    LOG( "- rep:  " << rep );
    arma::Row<DTYPE> median(dense);
    arma::Row<DTYPE> mean(dense);
    arma::Row<DTYPE> qhigh(dense);
    arma::Row<DTYPE> qlow(dense);

    xval::KFoldCV<MODEL,METRIC,arma::Mat<DTYPE>,arma::Row<size_t>> 
                                    xv(fold,trainset.inputs_,trainset.labels_,trainset.num_class_,true);
    for (size_t i=0;i<dense;i++)
    {
      auto res = xv.TrainAndEvaluate(ls[i]);
      arma::Row<DTYPE> q = {0.01, 0.5, 0.99};
      arma::Row<DTYPE> qs = arma::quantile(res,q);
      mean[i] = arma::mean(res);
      qlow[i] = qs[0];
      median[i] = qs[1];
      qhigh[i] = qs[2];
    }

    MODEL model_mean( trainset.inputs_,trainset.labels_,trainset.num_class_,ls[mean.index_max()]);  
    MODEL model_med(  trainset.inputs_,trainset.labels_,trainset.num_class_,ls[median.index_max()]);
    MODEL model_qlow( trainset.inputs_,trainset.labels_,trainset.num_class_,ls[qlow.index_max()]);  
    MODEL model_qhigh(trainset.inputs_,trainset.labels_,trainset.num_class_,ls[qhigh.index_max()]); 

    METRIC  metric;
    mean_sel[rep]  = metric.Evaluate(model_mean,  testset.inputs_,testset.labels_);
    med_sel[rep]   = metric.Evaluate(model_med,   testset.inputs_,testset.labels_);
    qlow_sel[rep]  = metric.Evaluate(model_qlow,  testset.inputs_,testset.labels_);
    qhigh_sel[rep] = metric.Evaluate(model_qhigh, testset.inputs_,testset.labels_);

  }
  mean_sel.save("mean.csv",arma::csv_ascii);
  med_sel.save("med.csv",arma::csv_ascii);
  qlow_sel.save("qlow.csv",arma::csv_ascii);
  qhigh_sel.save("qhigh.csv",arma::csv_ascii);
}

// multiple-cross validation procedure for one given model
int main ( int argc, char** argv )
{

  size_t id = 11; // 11,37,39,53,61
  size_t fold = 30;
  size_t reps = 100;
  size_t scale = 0;
  size_t dense = 1000;
  std::string what = "lda";

  // Parse command-line arguments
  for (int i = 1; i < argc; ++i) 
  {
      std::string arg = argv[i];

      if (arg == "-id" && i + 1 < argc) 
      {
        id = std::stoi(argv[i + 1]); // Convert string to int
        i++; // Skip next argument as it is already processed
      }
      else if (arg == "-nfold" && i + 1 < argc) 
      {
        fold = std::stoi(argv[i + 1]); // Convert string to int
        i++; // Skip next argument as it is already processed
      }
      else if (arg == "-reps" && i + 1 < argc) 
      {
        reps = std::stoi(argv[i + 1]); // Convert string to int
        i++; // Skip next argument as it is already processed
      }
      else if (arg == "-scale" && i + 1 < argc) 
      {
        scale = std::stoi(argv[i + 1]); // Convert string to int
      }
      else if (arg == "-what" && i + 1 < argc) 
      {
        what = argv[i + 1]; // Convert string to int
      }
      else if (arg == "-dense" && i + 1 < argc) 
      {
        dense = std::stoi(argv[i + 1]); // Convert string to int
      }
  }
  
  PRINT_VAR(id);
  PRINT_VAR(fold);
  PRINT_VAR(reps);
  PRINT_VAR(scale);
  PRINT_VAR(dense);

  std::filesystem::path TODAY = "12_03_25";
  std::filesystem::path dir = EXP_PATH/TODAY;
  if (scale)
    dir = dir / std::string("trans_"+std::to_string(id)+"_"+what+std::to_string(fold));
  else
    dir = dir / std::string(std::to_string(id)+"_"+what+std::to_string(fold));

  std::filesystem::create_directories(dir);
  std::filesystem::current_path(dir);

  // generate dataset (small_datasets)
  OpenML dataset(id);
  OpenML trainset,testset;

  data::StratifiedSplit(dataset,trainset,testset,0.2);
  /* data::Split(dataset,trainset,testset,0.2); */

  data::regression::Transformer<mlpack::data::StandardScaler,
                                OpenML > trans(trainset);
  if (scale)
  {
    trainset = trans.TransInp(trainset);
    testset = trans.TransInp(testset);
  }

  arma::Row<DTYPE> ls = arma::logspace<arma::Row<DTYPE>>(-10,4,dense);
  srowvec ns = arma::regspace<srowvec>(1,1000,dense);

  if ( what == "lda" )
    RunExperiment<LDA,OpenML,rowvec>(trainset,testset,fold,dense,reps,ls);
  else if ( what == "nmc" )
    RunExperiment<NMC,OpenML,rowvec>(trainset,testset,fold,dense,reps,ls);
  else if ( what == "qda" )
    RunExperiment<QDA,OpenML,rowvec>(trainset,testset,fold,dense,reps,ls);
  else if ( what == "lreg" )
    RunExperiment<LREG,OpenML,rowvec>(trainset,testset,fold,dense,reps,ls);
  else if ( what == "lsvc" )
    RunExperiment<LSVC,OpenML,rowvec>(trainset,testset,fold,dense,reps,ls);
  else if ( what == "gsvc" )
    RunExperiment<GSVC,OpenML,rowvec>(trainset,testset,fold,dense,reps,ls);
  else if ( what == "adab" )
    RunExperiment<ADAB,OpenML,srowvec>(trainset,testset,fold,dense,reps,ns);
  else if ( what == "dt" )
    RunExperiment<DT,OpenML,srowvec>(trainset,testset,fold,dense,reps,ns);
  else if ( what == "rfor" )
    RunExperiment<RFOR,OpenML,srowvec>(trainset,testset,fold,dense,reps,ns);
  else if ( what == "nnc" )
    RunExperiment<NNC,OpenML,srowvec>(trainset,testset,fold,dense,reps,ns);
  else
    ERR("NOTHING TO RUN");

  return 0;
  
};

