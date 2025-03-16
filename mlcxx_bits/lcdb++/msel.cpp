/**
 * @file msel.cpp
 * @author Ozgur Taylan Turan
 *
 * trying to select models here.
 */

#include <headers.h>

using MyData = data::regression::Dataset<>;
using OpenML = data::oml::Dataset<size_t>;
/* using LR = mlpack::LinearRegression<>;  // x */
/* using LREG = mlpack::LogisticRegression<>;  // x */
using MODEL = algo::classification::LDC<>; const std::string NAME = "lda";  // x
/* using MODEL = algo::classification::QDC<>; const std::string NAME = "qda";  // x */
/* using MODEL = mlpack::NaiveBayes<>; const std::string NAME = "nb";  // x */
/* using MODEL = mlpack::LinearSVM<>; const std::string NAME = "lsvm";  // x */
using METRIC = mlpack::Accuracy;

/* using MODEL = algo::classification::QDC<>;  // x */
/* using MODEL = mlpack::AdaBoost<mlpack::ID3DecisionStump>; */ 

/* using MODEL = mlpack::AdaBoost<mlpack::ID3DecisionStump>;using METRIC = mlpack::Accuracy; const std::string NAME = "adab"; */

/* using MODEL = algo::classification::LogisticRegression<>;  // x */
/* using MODEL = algo::regression::KernelRidge<mlpack::GaussianKernel>;  // x */
/* using MODEL = mlpack::LinearRegression< >;  // x */

arma::uword find ( const arma::vec &data, double value )
{
	return arma::index_min(arma::abs(data - value));
}

// multiple-cross validation procedure for one given model
int main ( int argc, char** argv )
{

  size_t id; 
  size_t fold;
  size_t reps;

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

  }

  PRINT_VAR(fold)
  std::filesystem::path TODAY = "13_03_25";
  /* std::filesystem::path dir = EXP_PATH/"02_03_25"/std::string("strat_"+std::to_string(id)+"_lreg_5"); */
  /* std::filesystem::path dir = EXP_PATH/TODAY/std::string("strat_"+std::to_string(id)+"_"+NAME+"_5"); */
  std::filesystem::path dir = EXP_PATH/TODAY/std::string(std::to_string(id)+"_"+NAME+std::to_string(fold));
  std::filesystem::create_directories(dir);
  std::filesystem::current_path(dir);

  // generate dataset (small_datasets)
  OpenML dataset(id);
  OpenML trainset,testset;

  

  data::StratifiedSplit(dataset,trainset,testset,0.2);
  /* data::Split(dataset,trainset,testset,0.2); */

  /* data::regression::Transformer<mlpack::data::StandardScaler, */
  /*                               OpenML > trans(trainset); */
  /* trainset = trans.TransInp(trainset); */
  /* testset = trans.TransInp(testset); */
  trainset = dataset;
  testset = dataset;

  size_t dense = 1000;
  arma::Row<DTYPE> ls = arma::logspace<arma::Row<DTYPE>>(-10,4,dense);
  /* auto ls = arma::regspace<arma::Row<size_t>>(0,10,dense); */
  /* LR mopt(trainset.inputs_,trainset.labels_,lopt); */
  /* PRINT_VAR(mopt.ComputeError(testset.inputs_,testset.labels_)); */
  /* size_t reps = 1000; */
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
                                    xv(fold,trainset.inputs_,trainset.labels_,true);
    for (size_t i=0;i<dense;i++)
    {
      /* LOG( "----i:  " << i); */
      auto res = xv.TrainAndEvaluate(2,ls[i]);
      arma::Row<DTYPE> q = {0.4, 0.5, 0.6};
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

    /* MODEL model_mean( trainset.inputs_,trainset.labels_,ls[mean.index_max()]); */  
    /* MODEL model_med(  trainset.inputs_,trainset.labels_,ls[median.index_max()]); */
    /* MODEL model_qlow( trainset.inputs_,trainset.labels_,ls[qlow.index_max()]); */  
    /* MODEL model_qhigh(trainset.inputs_,trainset.labels_,ls[qhigh.index_max()]); */ 


    METRIC  metric;
    mean_sel[rep]  = metric.Evaluate(model_mean,  testset.inputs_,testset.labels_);
    med_sel[rep]   = metric.Evaluate(model_med,   testset.inputs_,testset.labels_);
    qlow_sel[rep]  = metric.Evaluate(model_qlow,  testset.inputs_,testset.labels_);
    qhigh_sel[rep] = metric.Evaluate(model_qhigh, testset.inputs_,testset.labels_);

    /* MODEL model_mean( trainset.inputs_,trainset.labels_,ls[mean.index_min()]); */  
    /* MODEL model_med(  trainset.inputs_,trainset.labels_,ls[median.index_min()]); */
    /* MODEL model_qlow( trainset.inputs_,trainset.labels_,ls[qlow.index_min()]); */  
    /* MODEL model_qhigh(trainset.inputs_,trainset.labels_,ls[qhigh.index_min()]); */ 

    /* mean_sel[rep]  = model_mean.ComputeError(testset.inputs_,testset.labels_); */
    /* med_sel[rep]   = model_med.ComputeError(testset.inputs_,testset.labels_); */
    /* qlow_sel[rep]  = model_qlow.ComputeError(testset.inputs_,testset.labels_); */
    /* qhigh_sel[rep] = model_qhigh.ComputeError(testset.inputs_,testset.labels_); */

  }
  mean_sel.save("mean.csv",arma::csv_ascii);
  med_sel.save("med.csv",arma::csv_ascii);
  qlow_sel.save("qlow.csv",arma::csv_ascii);
  qhigh_sel.save("qhigh.csv",arma::csv_ascii);
  return 0;
};

/* int main() */
/* { */
/* 	arma::vec data = {1.0, 3.5, 7.2, 10.8, 15.3}; */
/* 	arma::vec values = {2.0, 9.0, 16.0}; */

/* 	arma::uvec indices = find_nearest_indices(data, values); */

/* 	std::cout << "Indices of nearest values: " << indices.t() << std::endl; */

/* 	return 0; */
/
