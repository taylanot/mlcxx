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

  /* data::classification::oml::Dataset dataset(1461); */
  data::classification::oml::Dataset dataset(1063);
  /* data::classification::oml::Dataset dataset(1063); */
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

  data::StratifiedSplit(dataset,trainset,testset,0.5);
  PRINT(dataset.size_);
  PRINT(dataset.dimension_);
  PRINT(dataset.num_class_);
  size_t repeat = 1000;
  arma::irowvec Ns = arma::regspace<arma::irowvec>(1,1,100);

  /* arma::irowvec Ns = {1}; */
  /* arma::irowvec Ns = arma::regspace<arma::irowvec>(10,1,10); */

  /* algo::classification::LogisticRegression<> model(trainset.inputs_,trainset.labels_,3); */
  /* mlpack::LogisticRegression<> model(trainset.inputs_.n_rows,0); */
  /* model.Train(trainset.inputs_,trainset.labels_); */

  /* PRINT_VAR(model.ComputeAccuracy(trainset.inputs_,trainset.labels_)); */

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
  /* src::LCurve<algo::classification::QDC<>,mlpack::Accuracy> lcurve(Ns,repeat,true,true); */
  /* src::LCurve<algo::classification::QDC<>,utils::BrierLoss> lcurve(Ns,repeat,true,true); */
  /* src::LCurve<algo::classification::QDC<>,utils::AUC> lcurve(Ns,repeat,true,true); */
  /* src::LCurve<algo::classification::SVM<mlpack::GaussianKernel,0>,mlpack::Accuracy> lcurve(Ns,repeat,true,true); */
  /* src::LCurve<algo::classification::SVM<>,mlpack::Accuracy> lcurve(Ns,repeat,true,true); */
  /* src::LCurve<algo::classification::NMC<>,mlpack::Accuracy> lcurve(Ns,repeat,true,true); */
  /* src::LCurve<algo::classification::NNC<>,mlpack::Accuracy> lcurve(Ns,repeat,true,false,true); */
  /* src::LCurve<algo::classification::LogisticRegression<>,mlpack::Accuracy> lcurve(Ns,repeat,true,true); */
  /* src::LCurve<algo::classification::LogisticRegression<>,utils::CrossEntropy> lcurve(Ns,repeat,false,true); */
  /* src::LCurve<algo::classification::LogisticRegression<>,utils::BrierLoss> lcurve(Ns,repeat,true,true); */
  /* src::LCurve<algo::classification::OnevAll<mlpack::LogisticRegression<>>,mlpack::Accuracy> lcurve(Ns,repeat,false,true); */
  /* src::LCurve<algo::classification::SVM<>,utils::CrossEntropy> lcurve(Ns,repeat,true,true); */
  /* src::LCurve<algo::classification::SVM<mlpack::EpanechnikovKernel>,mlpack::Accuracy> lcurve(Ns,repeat,true,true); */
  /* src::LCurve<algo::classification::SVM<mlpack::GaussianKernel>,mlpack::Accuracy> lcurve(Ns,repeat,true,true); */
  /* src::LCurve<algo::classification::LDC<>,mlpack::Accuracy> lcurve(Ns,repeat,true,true); */
  /* src::LCurve<algo::classification::LogisticRegression<>,mlpack::Accuracy> lcurve(Ns,repeat,false,true); */
  /* src::LCurve<mlpack::RandomForest<>,utils::BrierLoss> lcurve(Ns,repeat,true,true); */
  /* src::LCurve<mlpack::RandomForest<>,mlpack::Accuracy> lcurve(Ns,repeat,true,true); */
  /* src::LCurve<algo::classification::SVM<>,utils::LogLoss> lcurve(Ns,repeat,true,false,true); */
  /* src::LCurve<algo::classification::QDC<>,mlpack::Accuracy> lcurve(Ns,repeat,true,true); */
  /* src::LCurve<algo::classification::LDC<>,utils::LogLoss> lcurve(Ns,repeat,false,false,true); */
  /* src::LCurve<algo::classification::NNC<>,mlpack::Accuracy> lcurve(Ns,repeat,true,true); */
  /* src::LCurve<mlpack::RandomForest<>,mlpack::Accuracy> lcurve(Ns,repeat,true,true); */
  /* src::LCurve<algo::classification::NMC<>,mlpack::Accuracy> lcurve(Ns,repeat,true,false,true); */
  /* src::LCurve<mlpack::AdaBoost<mlpack::ID3DecisionStump>,mlpack::Accuracy> lcurve(Ns,repeat,true,true); */
  /* src::LCurve<mlpack::NaiveBayesClassifier<>,utils::CrossEntropy> lcurve(Ns,repeat,true,true); */
  /* src::LCurve<mlpack::RandomForest<>,utils::AUC> lcurve(Ns,repeat,true,true); */
  /* src::LCurve<algo::classification::NMC<>,mlpack::Accuracy> lcurve(Ns,repeat,false,true); */
  src::LCurve<algo::classification::NMC<>,utils::AUC> lcurve(Ns,repeat,false,true);
  /* src::LCurve<mlpack::RandomForest<>,utils::CrossEntropy> lcurve(Ns,repeat,true,true); */
  /* src::LCurve<mlpack::RandomForest<>,mlpack::Accuracy> lcurve(Ns,repeat,true,true); */
  /* lcurve.Split(trainset,testset,arma::unique(dataset.labels_).eval().n_elem); */
  lcurve.RandomSet(dataset,arma::unique(dataset.labels_).eval().n_elem);
  /* lcurve.Additive(dataset,arma::unique(dataset.labels_).eval().n_elem); */
  PRINT(lcurve.GetResults());
  /* lcurve.Bootstrap(dataset.inputs_,dataset.labels_,arma::unique(dataset.labels_).eval().n_elem); */
  /* lcurve.Split(trainset,testset,2,1.e-6); */
  /* lcurve.Split(trainset,testset,arma::unique(dataset.labels_).eval().n_elem,1e-8); */
  /* lcurve.Bootstrap(trainset.inputs_,trainset.labels_); */
  /* src::LCurveHPT<algo::classification::LDC<>,mlpack::Accuracy> lcurve(Ns,repeat,0.2,false,true); */
  /* auto lambdas = arma::logspace<arma::Row<DTYPE>>(-2,1,10); */
  /* lcurve.Bootstrap(trainset.inputs_,trainset.labels_,mlpack::Fixed(arma::unique(dataset.labels_).eval().n_elem),lambdas); */
  /* lcurve.Bootstrap(dataset.inputs_,dataset.labels_,arma::unique(dataset.labels_).eval().n_elem,1.); */
  /* lcurve.Bootstrap(trainset.inputs_,trainset.labels_,size_t(3),0.1); */
  /* lcurve.Bootstrap(trainset.inputs_,trainset.labels_,size_t(3),0.1); */
  /* PRINT(arma::unique(dataset.labels_).eval().n_elem); */

  /* src::LCurveHPT<algo::classification::SVM<>,mlpack::Accuracy> lcurve(Ns,repeat,0.2,true,false,true); */
  /* lcurve.Split(trainset,testset,mlpack::Fixed(3),arma::logspace(-2,2,10)); */
  /* lcurve.Split(trainset,testset,4,3); */
  /* lcurve.Split(trainset_,testset_,1e-6); */
  /* PRINT(lcurve.GetResults().has_nan()); */
  lcurve.GetResults().save("small-heyhey2_.csv",arma::csv_ascii);

  PRINT_TIME(timer.toc());

  
  return 0;
}
