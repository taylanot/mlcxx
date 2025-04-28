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
typedef mlpack::FFN<mlpack::CrossEntropyError,mlpack::HeInitialization> Network;
using ANN = algo::ANN<Network> ;
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
/*   size_t did = 1046;size_t rep = 100; */
/*   std::filesystem::path dir = EXP_PATH/"17_12_24/mxval"/(std::to_string(did))/"ks/lreg-notscale"; */
/*   std::filesystem::create_directories(dir); */
/*   OpenML dataset(did); */
/*   /1* data::classification::Transformer scale(dataset); *1/ */
/*   /1* dataset = scale.Trans(dataset); *1/ */

/*   arma::Row<size_t> ks = {2,5,10,20}; */
/*   for (size_t i=0; i<ks.n_elem; i++) */
/*     std::filesystem::create_directories(dir/(std::to_string(ks[i]))); */

/*   #pragma omp parallel for collapse(2) */
/*   for (size_t i=0; i<ks.n_elem; i++) */
/*     for (size_t j=0; j<rep; j++) */
/*     { */
/*       LOG("ks:"<<ks[i]); */
/*       xval::KFoldCV<LREG, mlpack::Accuracy> xv(ks[i], dataset.inputs_, dataset.labels_,true); */
/*       xv.TrainAndEvaluate(dataset.num_class_,0.).save(dir/(std::to_string(ks[i]))/(std::to_string(j)+".csv"),arma::csv_ascii); */
/*     } */
/* } */

/* int main ( int argc, char** argv ) */
/* { */
/*   size_t did = 1046; */
/*   std::filesystem::path dir = EXP_PATH/"17_12_24/mxval"/(std::to_string(did))/"ks/lreg"; */
/*   std::filesystem::create_directories(dir); */
/*   OpenML dataset(did); */

/*   arma::Row<size_t> ks = {2,5,10,20}; */
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

int main ( int argc, char** argv )
{
  size_t did = 1046;size_t rep = 1000;
  std::filesystem::path dir = EXP_PATH/"02_01_25/mxval"/(std::to_string(did))/"ks/ldc-hpt-noscale";
  std::filesystem::create_directories(dir);
  OpenML dataset(did);
  /* data::classification::Transformer scale(dataset); */
  /* dataset = scale.Trans(dataset); */

  arma::Row<size_t> ks = {2};
  arma::Row<DTYPE> ls = arma::logspace<arma::Row<DTYPE>>(-3,0,10);

  for (size_t i=0; i<ks.n_elem; i++)
    for (size_t j=0; j<ls.n_elem; j++)
      std::filesystem::create_directories(dir/(std::to_string(ks[i]))/(std::to_string(j)));

  #pragma omp parallel for collapse(3)
  for (size_t i=0; i<ks.n_elem; i++)
    for (size_t j=0; j<rep; j++)
      for (size_t k=0; k<ls.n_elem; k++)
    {
      LOG("ks:  "<<ks[i]<<"ls:  "<<ls[k]);
      xval::KFoldCV<LDC, mlpack::Accuracy> xv(ks[i], dataset.inputs_, dataset.labels_,true);
      xv.TrainAndEvaluate(dataset.num_class_,ls[k]).save(dir/(std::to_string(ks[i]))/(std::to_string(k))/(std::to_string(j)+".csv"),arma::csv_ascii);
    }
}


