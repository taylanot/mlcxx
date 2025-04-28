/**
 * @file xval2.cpp
 * @author Ozgur Taylan Turan
 *
 * Let's check xval results
 */

#include <headers.h>

using OpenML = data::classification::oml::Dataset<>;
using Network = mlpack::FFN<mlpack::CrossEntropyError,
                            mlpack::HeInitialization,arma::Mat<DTYPE>>;
using CN = algo::ANN<Network>;

void construct ( Network& network, const size_t nclass,
                 const size_t width = 16,
                 const size_t depth = 5 )
{
  for (size_t i=0; i<depth; i++)
  {
    network.Add<mlpack::Linear>(width); 
    network.Add<mlpack::ReLU>();                       
  }
  network.Add<mlpack::Linear>(nclass);
  network.Add<mlpack::Softmax>();
}

/* // multiple-cross validation procedure for one given model */
/* int main ( int argc, char** argv ) */
/* { */
/*   size_t did = 1046; size_t rep = 100;bool shuffle=true; */
/*   OpenML dataset(did); */

/*   std::filesystem::path dir = EXP_PATH/"31_12_24/mxval"/(std::to_string(did))/"ks/ann-shuff"; */
/*   std::filesystem::create_directories(dir); */
    
  

/*   arma::Row<size_t> ks = {2,5,10,20}; */
/*   arma::Row<DTYPE> lrs = {0.001,0.01, 0.1}; */

/*   for (size_t i=0; i<ks.n_elem; i++) */
/*     for (size_t j=0; j<lrs.n_elem; j++) */
/*       std::filesystem::create_directories(dir/(std::to_string(ks[i]))/("lr:"+std::to_string(lrs[j]))); */
/* Network network; construct(network,arma::unique(dataset.labels_).eval().n_elem); */
/*   /1* /2* algo::ANN<Network> model(dataset.inputs_, dataset.labels_,&network,false); *2/ *1/ */
/*   /1* algo::ANN<Network> model; *1/ */
/*   /1* model.Train(dataset.inputs_,dataset.labels_,&network,false); *1/ */

/*   /1* arma::Row<size_t> pred; *1/ */
/*   /1* model.Classify(dataset.inputs_,pred); *1/ */
/*   /1* PRINT(pred.cols(0,5)); *1/ */
/*   /1* PRINT(dataset.labels_.cols(0,5)); *1/ */

/*   #pragma omp parallel for collapse(3) */
/*   for (size_t i=0; i<ks.n_elem; i++) */
/*     for (size_t j=0; j<rep; j++) */
/*       for (size_t l=0; l<lrs.n_elem; l++) */
/*       { */
/*         LOG("ks:"<<ks[i]<<"rep:"<<j); */
/*         xval::KFoldCV<CN,mlpack::Accuracy,arma::Mat<DTYPE>,arma::Row<size_t>> */ 
/*                                           xv(ks[i],dataset.inputs_,dataset.labels_,shuffle); */
/*         xv.TrainAndEvaluate(network,false,lrs[l],32).save(dir/(std::to_string(ks[i]))/("lr:"+std::to_string(lrs[l]))/(std::to_string(j)+".csv"),arma::csv_ascii); */
/*       } */

/* } */

using RFOR  = mlpack::RandomForest<>; 
using DT    = mlpack::DecisionTree<>; 
using NB    = mlpack::NaiveBayesClassifier<>;
using LDC   = algo::classification::LDC<>; 
using LREG  = algo::classification::LogisticRegression<>;  // x
using NMC   = algo::classification::NMC<>; 
using NNC   = algo::classification::NNC<>; 
using QDC   = algo::classification::QDC<>; 
using LSVC  = algo::classification::SVM<mlpack::LinearKernel>; 
using GSVC  = algo::classification::SVM<mlpack::GaussianKernel>; 
using ADAB  = mlpack::AdaBoost<mlpack::ID3DecisionStump>; 
/* // multiple-cross validation procedure for one given model */
/* int main ( int argc, char** argv ) */
/* { */
/*   size_t did = 1046; size_t rep = 100; bool shuffle=true; */
/*   OpenML dataset(did,"omldatasets"); */

/*   std::filesystem::path dir = "23_01_25_add/mxval/varvsk/dt"; */
/*   dir = dir/(std::to_string(did)); */
/*   std::filesystem::create_directories(dir); */
/*   size_t N = dataset.size_; */
/*   size_t C = dataset.num_class_; */
/*   arma::Row<size_t> ks = {2,5,10}; */
/*   #pragma omp parallel for collapse(2) */
/*   for (size_t i=0; i<ks.n_elem; i++) */
/*     for (size_t j=0; j<rep; j++) */
/*     { */
/*       LOG("ks:"<<ks[i]<<"rep:"<<j); */
/*       xval::KFoldCV<DT,mlpack::Accuracy,arma::Mat<DTYPE>,arma::Row<size_t>> */ 
/*               xv(ks[i],dataset.inputs_,dataset.labels_,C,shuffle); */
/*       std::filesystem::create_directories(dir/std::to_string(j)); */
/*       xv.TrainAndEvaluate(C).save(dir/std::to_string(j)/(std::to_string(ks[i])+".csv"),arma::csv_ascii); */
/*     } */
/* } */

// multiple-cross validation procedure for one given model
int main ( int argc, char** argv )
{
  arma::Row<size_t> dids = {4534,151,41027,32,4538,11,15,23,37};
  for (size_t k=0; k<dids.n_elem; k++)
  {
  size_t did = dids[k]; size_t rep = 100; bool shuffle=true;
  LOG("ID:"<<did);
  OpenML dataset(did);
  std::filesystem::path dir = EXP_PATH/"23_01_25/xval_all/nnc";
  dir = dir/(std::to_string(did));
  std::filesystem::create_directories(dir);
  size_t N = dataset.size_;
  size_t C = dataset.num_class_;
  arma::Row<size_t> ks = {2,5,10,20};
  #pragma omp parallel for collapse(2)
  for (size_t i=0; i<ks.n_elem; i++)
    for (size_t j=0; j<rep; j++)
    {
      xval::KFoldCV<LREG,mlpack::Accuracy,arma::Mat<DTYPE>,arma::Row<size_t>> 
              /* xv(ks[i],dataset.inputs_,dataset.labels_,C,shuffle); */
              xv(ks[i],dataset.inputs_,dataset.labels_,shuffle);
      std::filesystem::create_directories(dir/std::to_string(j));
      xv.TrainAndEvaluate(C).save(dir/std::to_string(j)/(std::to_string(ks[i])+".csv"),arma::csv_ascii);
    }
  }
}
