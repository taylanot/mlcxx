/**
 * @file xval2.cpp
 * @author Ozgur Taylan Turan
 *
 * Let's check xval results
 */

#include <headers.h>

using MyData = data::regression::Dataset<>;
using OpenML = data::oml::Dataset<DTYPE>;
/* using LR = mlpack::LinearRegression<>;  // x */
/* using LREG = mlpack::LogisticRegression<>;  // x */
/* using MODEL = algo::classification::LDC<>;  // x */
/* using MODEL = algo::classification::QDC<>;  // x */
/* using MODEL = algo::classification::QDC<>;  // x */
/* using MODEL = mlpack::AdaBoost<mlpack::ID3DecisionStump>; */ 
/* using MODEL = algo::classification::LogisticRegression<>;  // x */
/* using MODEL = algo::regression::KernelRidge<mlpack::GaussianKernel>;  // x */
using MODEL = mlpack::LinearRegression< >;  // x


/* // multiple-cross validation procedure for one given model */
/* int main ( int argc, char** argv ) */
/* { */
/*   std::filesystem::path dir = EXP_PATH/"26_02_25"/"errdist1d_100_20fold"; */
/*   std::filesystem::create_directories(dir); */
/*   std::filesystem::current_path(dir); */

/*   // generate dataset */ 
/*   size_t D = 1; */
/*   double sig = 1.; */
/*   MyData trainset(D,100); */
/*   MyData testset(D,10000); */
/*   trainset.Generate(std::string("Linear"),sig); */
/*   testset.Generate(std::string("Linear"),sig); */
/*   double lopt = double(D) * std::pow(sig,2) / */ 
/*                     std::pow(arma::norm(arma::ones<arma::Row<DTYPE>>(D),2),2); */

/*   size_t dense = 10000; */
/*   arma::Row<DTYPE> ls = arma::linspace<arma::Row<DTYPE>>(0,5,dense); */
/*   /1* LR mopt(trainset.inputs_,trainset.labels_,lopt); *1/ */
/*   /1* PRINT_VAR(mopt.ComputeError(testset.inputs_,testset.labels_)); *1/ */
/*   size_t reps = 100; */
/*   arma::Row<DTYPE> mean_sel(reps); */
/*   arma::Row<DTYPE> med_sel(reps); */
/*   arma::Row<DTYPE> qlow_sel(reps); */
/*   arma::Row<DTYPE> qhigh_sel(reps); */

/*   #pragma omp parallel for */
/*   for (size_t rep=0; rep < reps; rep++) */
/*   { */
/*     LOG( "rep:  " << rep ); */
/*     arma::Row<DTYPE> median(dense); */
/*     arma::Row<DTYPE> mean(dense); */
/*     arma::Row<DTYPE> qhigh(dense); */
/*     arma::Row<DTYPE> qlow(dense); */

/*     xval::KFoldCV<LR,mlpack::MSE,arma::Mat<DTYPE>,arma::Row<DTYPE>> */ 
/*                                     xv(20,trainset.inputs_,trainset.labels_,true); */
/*     for (size_t i=0;i<dense;i++) */
/*     { */
/*       auto res = xv.TrainAndEvaluate(ls[i],true); */
/*       arma::Row<DTYPE> q = {0.05, 0.5, 0.95}; */
/*       arma::Row<DTYPE> qs = arma::quantile(res,q); */
/*       mean[i] = arma::mean(res); */
/*       qlow[i] = qs[0]; */
/*       median[i] = qs[1]; */
/*       qhigh[i] = qs[2]; */
/*     } */
/*     /1* mean_sel[rep]  = (ls[mean.index_min()]); *1/ */
/*     /1* med_sel[rep]   = (ls[median.index_min()]); *1/ */
/*     /1* qlow_sel[rep]  = (ls[qlow.index_min()]); *1/ */
/*     /1* qhigh_sel[rep] = (ls[qhigh.index_min()]); *1/ */

/*     LR model_mean( trainset.inputs_,trainset.labels_ ,ls[mean.index_min()]); */  
/*     LR model_med(  trainset.inputs_,trainset.labels_ ,ls[median.index_min()]); */
/*     LR model_qlow( trainset.inputs_,trainset.labels_ ,ls[qlow.index_min()]); */  
/*     LR model_qhigh(trainset.inputs_,trainset.labels_ ,ls[qhigh.index_min()]); */ 

/*     mean_sel[rep]  = model_mean.ComputeError(testset.inputs_,testset.labels_); */
/*     med_sel[rep]   = model_med.ComputeError(testset.inputs_,testset.labels_); */
/*     qlow_sel[rep]  = model_qlow.ComputeError(testset.inputs_,testset.labels_); */
/*     qhigh_sel[rep] = model_qhigh.ComputeError(testset.inputs_,testset.labels_); */


/*   } */
/*   mean_sel.save("mean.csv",arma::csv_ascii); */
/*   med_sel.save("med.csv",arma::csv_ascii); */
/*   qlow_sel.save("qlow.csv",arma::csv_ascii); */
/*   qhigh_sel.save("qhigh.csv",arma::csv_ascii); */
/*   return 0; */
/* }; */

// multiple-cross validation procedure for one given model
int main ( int argc, char** argv )
{

  size_t id=8; // 11,37,39,53,61
                //
  // Parse command-line argumennts
  for (int i = 1; i < argc; ++i) 
  {
    std::string arg = argv[i];
    if (arg == "-id" && i + 1 < argc) 
    {
      id = std::stoi(argv[i + 1]); // Convert string to int
      break;
    }
  }

  std::filesystem::path TODAY = "04_03_25";
  /* std::filesystem::path dir = EXP_PATH/"02_03_25"/std::string("strat_"+std::to_string(id)+"_lreg_5"); */
  std::filesystem::path dir = EXP_PATH/TODAY/std::string("scl"+std::to_string(id)+"_lr_5");
  std::filesystem::create_directories(dir);
  std::filesystem::current_path(dir);

  // generate dataset (small_datasets)
  OpenML dataset(id);
  OpenML trainset,testset;

  

  /* data::StratifiedSplit(dataset,trainset,testset,0.2); */
  data::Split(dataset,trainset,testset,0.2);

  data::regression::Transformer<mlpack::data::StandardScaler,
                                OpenML > trans(trainset);
  trainset = trans.TransInp(trainset);
  testset = trans.TransInp(testset);

  size_t dense = 1000;
  arma::Row<DTYPE> ls = arma::linspace<arma::Row<DTYPE>>(0,2,dense);
  /* LR mopt(trainset.inputs_,trainset.labels_,lopt); */
  /* PRINT_VAR(mopt.ComputeError(testset.inputs_,testset.labels_)); */
  size_t reps = 1000;
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

    xval::KFoldCV<MODEL,mlpack::MSE,arma::Mat<DTYPE>,arma::Row<DTYPE>> 
                                    xv(5,trainset.inputs_,trainset.labels_,true);

    for (size_t i=0;i<dense;i++)
    {
      /* LOG( "----i:  " << i); */
      /* auto res = xv.TrainAndEvaluate(trainset.num_class_,ls[i]); */
      auto res = xv.TrainAndEvaluate(ls[i]);
      arma::Row<DTYPE> q = {0.05, 0.5, 0.95};
      arma::Row<DTYPE> qs = arma::quantile(res,q);
      mean[i] = arma::mean(res);
      qlow[i] = qs[0];
      median[i] = qs[1];
      qhigh[i] = qs[2];
    }

    /* MODEL model_mean( trainset.inputs_,trainset.labels_,trainset.num_class_,ls[mean.index_max()]); */  
    /* MODEL model_med(  trainset.inputs_,trainset.labels_,trainset.num_class_,ls[median.index_max()]); */
    /* MODEL model_qlow( trainset.inputs_,trainset.labels_,trainset.num_class_,ls[qlow.index_max()]); */  
    /* MODEL model_qhigh(trainset.inputs_,trainset.labels_,trainset.num_class_,ls[qhigh.index_max()]); */ 

    /* mean_sel[rep]  = model_mean.ComputeAccuracy(testset.inputs_,testset.labels_); */
    /* med_sel[rep]   = model_med.ComputeAccuracy(testset.inputs_,testset.labels_); */
    /* qlow_sel[rep]  = model_qlow.ComputeAccuracy(testset.inputs_,testset.labels_); */
    /* qhigh_sel[rep] = model_qhigh.ComputeAccuracy(testset.inputs_,testset.labels_); */

    MODEL model_mean( trainset.inputs_,trainset.labels_,ls[mean.index_min()]);  
    MODEL model_med(  trainset.inputs_,trainset.labels_,ls[median.index_min()]);
    MODEL model_qlow( trainset.inputs_,trainset.labels_,ls[qlow.index_min()]);  
    MODEL model_qhigh(trainset.inputs_,trainset.labels_,ls[qhigh.index_min()]); 

    mean_sel[rep]  = model_mean.ComputeError(testset.inputs_,testset.labels_);
    med_sel[rep]   = model_med.ComputeError(testset.inputs_,testset.labels_);
    qlow_sel[rep]  = model_qlow.ComputeError(testset.inputs_,testset.labels_);
    qhigh_sel[rep] = model_qhigh.ComputeError(testset.inputs_,testset.labels_);


  }
  mean_sel.save("mean.csv",arma::csv_ascii);
  med_sel.save("med.csv",arma::csv_ascii);
  qlow_sel.save("qlow.csv",arma::csv_ascii);
  qhigh_sel.save("qhigh.csv",arma::csv_ascii);
  return 0;
};

