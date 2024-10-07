/**
 * @file nodd.cpp
 * @author Ozgur Taylan Turan
 *
 * This file is for showing double descent mitigation via excluisoin of principle components.
 *
 */
#define DTYPE double  

#include <headers.h>



int main ( int argc, char** argv )
{
  /* std::filesystem::path path = "30_08_24/oml/3"; */
  /* std::filesystem::create_directories(path); */
  /* std::filesystem::current_path(path); */

  arma::wall_clock timer;
  timer.tic();


  /* size_t id = 37; // iris dataset */
  /* getopenmldataset(id); */  
  /* arma::Mat<DTYPE> data; */
  /* std::filesystem::path file = DATASET_PATH/"openml"/(std::to_string(id)+".arff"); */
  /* mlpack::data::DatasetInfo info; */
  /* mlpack::data::Load(file.c_str(), data, info); */
  /* PRINT(data.size()); */

  /* PRINT(extractDefaultTargetValue(xmlData)); */

  /* PRINT(fetchMetadata(61)); */

  /* PRINT(findLabelARFF(file.c_str(),"'class'")); */
  /* PRINT(findLabelARFF(file.c_str(),extractDefaultTargetValue(fetchMetadata(id)))); */
  /* PRINT(extractDefaultTargetValue(fetchMetadata(id))); */
  /* mlpack::data::LoadARFF<DTYPE>(file,mat); */

  data::classification::oml::Dataset dataset(11);
  /* data::classification::oml::Dataset dataset(99); */
  /* data::classification::oml::Dataset dataset(3); */
  /* data::classification::oml::Collect study(99); */
  /* data::classification::oml::Dataset dataset; */

  /* for ( size_t i=0; i<study.size_; i++) */
  /* { */
  /*   dataset = study.GetNext(); */
  /*   PRINT(arma::size(dataset.inputs_)); */
  /* } */

  data::classification::oml::Dataset trainset,testset;
  data::StratifiedSplit(dataset,trainset,testset,0.2);

  /* data::classification::Dataset trainset(2,100,2); */
  /* data::classification::Dataset testset(2,1000,2); */
  /* trainset.Generate("Simple"); */
  /* testset.Generate("Simple"); */
  

  size_t repeat = 1000;
  PRINT(dataset.size_);
  PRINT(dataset.dimension_);
  arma::irowvec Ns = arma::regspace<arma::irowvec>(2,1,size_t(trainset.size_*0.3));
  /* arma::irowvec Ns = arma::regspace<arma::irowvec>(2,10,100); */
  /* src::LCurve<mlpack::LinearSVM<>,mlpack::Accuracy> lcurve(Ns,repeat,true,false,true); */
  /* src::LCurve<algo::classification::KernelSVM<>,mlpack::Accuracy> lcurve(Ns,repeat,true,false,true); */
  /* src::LCurve<algo::classification::KernelSVM<mlpack::GaussianKernel>, */
  /* src::LCurve<algo::classification::LDC<>, */
  /* src::LCurve<algo::classification::KernelSVM<mlpack::LinearKernel>, */
  /* src::LCurve<algo::classification::QDC<>, */
              /* mlpack::Accuracy> lcurve(Ns,repeat,true,false,true); */
  /* src::LCurve<algo::classification::KernelSVM<mlpack::GaussianKernel>,mlpack::Accuracy, */
  /* data::classification::Transformer<mlpack::data::StandardScaler,data::classification::oml::Dataset<>> trans(trainset); */
  /* auto trainset_ = trans.Trans(trainset); */
  /* auto testset_ = trans.Trans(testset); */
  /* src::LCurve<algo::classification::OnevAll< */
  /*   algo::classification::SVM<mlpack::GaussianKernel>>,mlpack::Accuracy, */
  /* src::LCurve<algo::classification::OnevAll<mlpack::NaiveBayesClassifier<>>,mlpack::Accuracy, */
  /*   data::N_StratSplit> lcurve(Ns,repeat,true,false,true); */
  /* src::LCurve<algo::classification::LDC<>,mlpack::Accuracy> lcurve(Ns,repeat,true,false,true); */
  /* src::LCurve<algo::classification::SVM<>,mlpack::Accuracy> lcurve(Ns,repeat,true,false,true); */
  src::LCurve<algo::classification::NNC<>,mlpack::Accuracy> lcurve(Ns,repeat,true,false,true);
  /* src::LCurve<algo::classification::SVM<>,utils::LogLoss> lcurve(Ns,repeat,true,false,true); */
  /* src::LCurve<algo::classification::QDC<>,mlpack::Accuracy> lcurve(Ns,repeat,false,false,true); */
  /* src::LCurve<algo::classification::LDC<>,utils::LogLoss> lcurve(Ns,repeat,false,false,true); */
  /* src::LCurve<algo::classification::NNC<>,mlpack::Accuracy> lcurve(Ns,repeat,true,false,true); */
  /* src::LCurve<mlpack::RandomForest<>,mlpack::Accuracy> lcurve(Ns,repeat,true,false,true); */
  /* src::LCurve<algo::classification::NMC<>,mlpack::Accuracy> lcurve(Ns,repeat,true,false,true); */
  /* src::LCurve<mlpack::AdaBoost<>,mlpack::Accuracy> lcurve(Ns,repeat,false,false,true); */
  /* lcurve.Split(trainset,testset,4); */
  /* lcurve.Split(trainset,testset,2,1.e-6); */
  /* lcurve.Split(trainset,testset,arma::unique(dataset.labels_).eval().n_elem,1e-8); */
  /* lcurve.Bootstrap(trainset.inputs_,trainset.labels_,arma::unique(dataset.labels_).eval().n_elem,1e-8); */
  /* lcurve.Bootstrap(trainset.inputs_,trainset.labels_,arma::unique(dataset.labels_).eval().n_elem); */
  /* lcurve.Bootstrap(trainset.inputs_,trainset.labels_,size_t(3),0.1); */
  /* lcurve.Bootstrap(trainset.inputs_,trainset.labels_,size_t(3),0.1); */
  /* PRINT(arma::unique(dataset.labels_).eval().n_elem); */

  /* src::LCurveHPT<algo::classification::SVM<>,mlpack::Accuracy> lcurve(Ns,repeat,0.2,true,false,true); */
  /* lcurve.Split(trainset,testset,mlpack::Fixed(3),arma::logspace(-2,2,10)); */
  lcurve.Split(trainset,testset,3,1.);
  /* lcurve.Split(trainset_,testset_,1e-6); */
  PRINT(lcurve.test_errors_.save("nnc.csv",arma::csv_ascii));

  PRINT_TIME(timer.toc());

  
  return 0;
}
