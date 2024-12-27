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
/*   std::filesystem::path dir = EXP_PATH/"17_12_24/mxval"/(std::to_string(did))/"ks/ann"; */
/*   std::filesystem::create_directories(dir); */
/*   Dataset dataset(did); */

/*   arma::Row<size_t> ks = {2,5,10}; */
/*   arma::Row<DTYPE>ls = arma::logspace<arma::Row<DTYPE>>(-10,2,10); */

/*   for (size_t i=0; i<ks.n_elem; i++) */
/*     std::filesystem::create_directories(dir/(std::to_string(ks[i]))); */

/*   size_t rep = 100; */
/*   #pragma omp parallel for collapse(2) */
/*   for (size_t i=0; i<ks.n_elem; i++) */
/*     for (size_t j=0; j<rep; j++) */
/*     { */
/*       LOG("ks :  "<<ks[i]); */
/*       xval::KFoldCV<LREG, mlpack::Accuracy> xv(ks[i], dataset.inputs_, dataset.labels_,true); */
/*       xv.TrainAndEvaluate(dataset.num_class_).save(dir/(std::to_string(ks[i]))/(std::to_string(j)+".csv"),arma::csv_ascii); */
/*     } */
/* } */

/* template<typename RowType, typename MatType> */
/* void OneHotEncoding(const RowType& labelsIn, */
/*                     MatType& output) */
/* { */
/*   arma::Row<size_t> ulabels = arma::unique(labelsIn); */
/*   std::unordered_map<typename MatType::elem_type, size_t> labelMap; */
/*   // Here we loop over the unique indeces and fill the labelMap */
/*   for (size_t i = 0; i < ulabels.n_elem ; ++i) */
/*       labelMap[i] = ulabels[i]; */

/*   // Resize output matrix to necessary size, and fill it with zeros. */
/*   output.zeros(ulabels.n_elem, labelsIn.n_elem); */
/*   // Fill ones in at the required places. */
/*   for (size_t i = 0; i < labelsIn.n_elem; ++i) */
/*     output(labelMap[labelsIn[i]], i) = 1; */
/* } */

/* template<class O=DTYPE> */
/* arma::Mat<O> OneHotEncode( const arma::Row<size_t>& labels, */
/*                            const arma::Row<size_t>& ulabels ) */
/* { */
/*   size_t i=0; */
/*   return arma::Mat<O>(ulabels.n_elem, labels.n_elem). */
/*     each_col( [&](arma::vec& col){col(ulabels[labels[i++]])=1.;} ); */
/* } */

/* template<class O=DTYPE> */
/* arma::Row<size_t> OneHotDecode( const arma::Mat<O>& labels, */
/*                                 const arma::Row<size_t>& ulabels ) */
/* { */
/*   return ulabels.cols(arma::index_max(labels,0)); */
/* } */

/* // multiple-cross validation procedure for one given model */
/* int main ( int argc, char** argv ) */
/* { */
/*   size_t did = 11; */
/*   OpenML dataset(did); */

/*   arma::mat oneHot, temp; */
/*   OneHotEncoding(dataset.labels_, oneHot); */
/*   PRINT_VAR(dataset.labels_.cols(0,5)); */
/*   PRINT_VAR(oneHot.cols(0,5)); */
/* } */

/* int main ( int argc, char** argv ) */
/* { */
/*   size_t did = 11; */
/*   OpenML dataset(did); */

/*   arma::mat oneHot, temp; */
/*   PRINT(dataset.labels_.cols(0,5)); */
/*   oneHot = OneHotEncode(dataset.labels_, arma::unique(dataset.labels_).eval()).cols(0,5); */
/*   PRINT(oneHot) */
/*   arma::Row<size_t> labels = OneHotDecode(oneHot, arma::unique(dataset.labels_)); */
/*   PRINT(labels) */
/*   /1* PRINT_VAR(dataset.labels_.cols(0,5)); *1/ */
/*   /1* PRINT_VAR(oneHot.cols(0,5)); *1/ */
/* } */

// multiple-cross validation procedure for one given model
int main ( int argc, char** argv )
{
  size_t did = 11;
  OpenML dataset(did);
  std::filesystem::path dir = EXP_PATH/"17_12_24/mxval"/(std::to_string(did))/"ks/ann";
  std::filesystem::create_directories(dir);

  typedef mlpack::FFN<mlpack::CrossEntropyError,mlpack::HeInitialization> Network;

  Network network;
  network.Add<mlpack::Linear>(16); 
  network.Add<mlpack::ReLU>();                       
  network.Add<mlpack::Linear>(3);  
  network.Add<mlpack::Softmax>();  


  algo::ANN<Network> model(dataset.inputs_, dataset.labels_,&network,false);
  arma::Row<size_t> pred;
  model.Classify(dataset.inputs_,pred);

  mlpack::Accuracy metric;
  PRINT(metric.Evaluate(model,dataset.inputs_,dataset.labels_));
  /* for (size_t i=0; i<ks.n_elem; i++) */
  /*   std::filesystem::create_directories(dir/(std::to_string(ks[i]))); */

  /* size_t rep = 10; */
  /* #pragma omp parallel for collapse(2) */
  /* for (size_t i=0; i<ks.n_elem; i++) */
  /*   for (size_t j=0; j<rep; j++) */
  /*   { */
  /*     LOG("ks :  "<<ks[i]); */
  /*     xval::KFoldCV<NN, mlpack::Accuracy> xv(ks[i], dataset.inputs_, dataset.labels_,true); */
  /*     xv.TrainAndEvaluate(&network).save(dir/(std::to_string(ks[i]))/(std::to_string(j)+".csv"),arma::csv_ascii); */
  /*   } */
}


