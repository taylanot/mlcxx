/**
 * @file xval.cpp
 * @author Ozgur Taylan Turan
 *
 * Let's check xval results
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

//-----------------------------------------------------------------------------
//   main : our main function no need to do anything here...
//-----------------------------------------------------------------------------
/* int main ( int argc, char** argv ) */
/* { */
/*   std::filesystem::path dir = EXP_PATH/"13_12_24/hpt-100k-dense/lsvc"; */
/*   std::filesystem::create_directories(dir); */


/*   arma::irowvec ids = {11,37,53,61}; */
/*   /1* arma::irowvec ids = {39}; *1/ */
/*   arma::vec ls = arma::logspace<arma::vec>(-3,2,10); */
/*   for (size_t i=0; i<ids.n_elem; i++) */
/*   { */
/*     Dataset dataset(ids[i]); */
/*     std::filesystem::create_directories(dir/(std::to_string(i))); */
/*     for (size_t j=0; j<ls.n_elem; j++) */
/*     { */
/*       xval::KFoldCV<LSVC, mlpack::Accuracy> xv(10, dataset.inputs_, dataset.labels_,true); */
/*       xv.TrainAndEvaluate(dataset.num_class_,ls[i]).save(dir/(std::to_string(i))/(std::to_string(j)+".csv"),arma::csv_ascii); */
/*     } */
/*   } */
/* } */

/* int main ( int argc, char** argv ) */
/* { */
/*   size_t did = 1046; */
/*   std::filesystem::path dir = EXP_PATH/"17_12_24"/(std::to_string(did))/"ks/lreg"; */
/*   std::filesystem::create_directories(dir); */
/*   Dataset dataset(did); */

/*   /1* arma::Row<size_t> ks = {2,5,10,20,100}; *1/ */
/*   arma::Row<size_t> ks = {1000}; */
/*   arma::Row<DTYPE>ls = arma::logspace<arma::Row<DTYPE>>(-10,2,10); */
/*   for (size_t i=0; i<ks.n_elem; i++) */
/*     std::filesystem::create_directories(dir/(std::to_string(ks[i]))); */

/*   #pragma omp parallel for collapse(2) */
/*   for (size_t i=0; i<ks.n_elem; i++) */
/*     for (size_t j=0; j<ls.n_elem; j++) */
/*     { */
/*       LOG("ks:"<<ks[i]<<"ls:"<<ls[j]); */
/*       xval::KFoldCV<LREG, mlpack::Accuracy> xv(ks[i], dataset.inputs_, dataset.labels_,true); */
/*       xv.TrainAndEvaluate(dataset.num_class_,ls[i]).save(dir/(std::to_string(ks[i]))/(std::to_string(j)+".csv"),arma::csv_ascii); */
/*     } */
/* } */


/* // multiple-cross validation procedure for one given model */
/* int main ( int argc, char** argv ) */
/* { */
/*   size_t did = 11; */
/*   std::filesystem::path dir = EXP_PATH/"17_12_24/mxval"/(std::to_string(did))/"ks/lreg"; */
/*   std::filesystem::create_directories(dir); */
/*   Dataset dataset(did); */

/*   arma::Row<size_t> ks = {2,5,10}; */
/*   arma::Row<DTYPE>ls = arma::logspace<arma::Row<DTYPE>>(-10,2,10); */

/*   for (size_t i=0; i<ks.n_elem; i++) */
/*     std::filesystem::create_directories(dir/(std::to_string(ks[i]))); */

/*   size_t rep = 1000; */
/*   #pragma omp parallel for collapse(2) */
/*   for (size_t i=0; i<ks.n_elem; i++) */
/*     for (size_t j=0; j<rep; j++) */
/*     { */
/*       LOG("ks :  "<<ks[i]); */
/*       xval::KFoldCV<LREG, mlpack::Accuracy> xv(ks[i], dataset.inputs_, dataset.labels_,true); */
/*       xv.TrainAndEvaluate(dataset.num_class_).save(dir/(std::to_string(ks[i]))/(std::to_string(j)+".csv"),arma::csv_ascii); */
/*     } */
/* } */

/* // multiple-cross validation procedure for one given model */
int main ( int argc, char** argv )
{
  Dataset dataset(2,5,2);
  dataset.Generate("Simple");

  mlpack::RandomSeed(SEED);
  xval::KFoldCV<LREG, mlpack::Accuracy> xv(2, dataset.inputs_, dataset.labels_,true);
  mlpack::RandomSeed(SEED);
  xval::KFoldCV<QDC, mlpack::Accuracy> xv2(2, dataset.inputs_, dataset.labels_,true);
  PRINT_VAR(xv.xs);
  PRINT_VAR(xv.GetTrainingSubset(xv.xs,0));
  PRINT_VAR(xv2.xs);
  PRINT_VAR(xv2.GetTrainingSubset(xv2.xs,0));
}

/* int main ( int argc, char** argv ) */
/* { */
/*   mlpack::RandomSeed(123); // Set a specific random seed. */
/*   PRINT_VAR(arma::randn(1)); */
/*   PRINT_VAR(arma::randn(1)); */
/*   mlpack::RandomSeed(124); // Set a specific random seed. */
/*   PRINT_VAR(arma::randn(1)); */
/*   PRINT_VAR(arma::randn(1)); */
/* } */
