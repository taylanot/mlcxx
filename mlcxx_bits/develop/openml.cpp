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
  arma::wall_clock timer;
  timer.tic();

  data::oml::Dataset dataset(451);

  arma::irowvec Ns = arma::regspace<arma::irowvec>(10,1,100);

  src::LCurveHPT<algo::classification::LDC<>,mlpack::Accuracy> lcurve(Ns,10,0.2,true,true);
  arma::Mat<DTYPE> Cs = arma::logspace(1e-8,1e-2,5);
  lcurve.Bootstrap(dataset,Cs);
  PRINT_VAR(arma::size(lcurve.GetResults()));

  PRINT_TIME(timer.toc());

  
  return 0;
}


/* int main ( int argc, char** argv ) */
/* { */
/*   /1* std::filesystem::path path = "30_08_24/oml/3"; *1/ */
/*   /1* std::filesystem::create_directories(path); *1/ */
/*   /1* std::filesystem::current_path(path); *1/ */

/*   arma::wall_clock timer; */
/*   timer.tic(); */


/*   /1* size_t id = 37; // iris dataset *1/ */
/*   /1* getopenmldataset(id); *1/ */  
/*   /1* arma::Mat<DTYPE> data; *1/ */
/*   /1* std::filesystem::path file = DATASET_PATH/"openml"/(std::to_string(id)+".arff"); *1/ */
/*   /1* mlpack::data::DatasetInfo info; *1/ */
/*   /1* mlpack::data::Load(file.c_str(), data, info); *1/ */
/*   /1* PRINT(data.size()); *1/ */

/*   /1* PRINT(extractDefaultTargetValue(xmlData)); *1/ */

/*   /1* PRINT(fetchMetadata(61)); *1/ */

/*   /1* PRINT(findLabelARFF(file.c_str(),"'class'")); *1/ */
/*   /1* PRINT(findLabelARFF(file.c_str(),extractDefaultTargetValue(fetchMetadata(id)))); *1/ */
/*   /1* PRINT(extractDefaultTargetValue(fetchMetadata(id))); *1/ */
/*   /1* mlpack::data::LoadARFF<DTYPE>(file,mat); *1/ */

/*   /1* data::classification::oml::Dataset dataset(1461); *1/ */
/*   data::oml::Dataset dataset(11); */
/*   /1* data::classification::oml::Dataset dataset(1063); *1/ */
/*   /1* data::classification::oml::Dataset dataset(99); *1/ */
/*   /1* data::classification::oml::Dataset dataset(3); *1/ */
/*   /1* data::classification::oml::Collect study(99); *1/ */
/*   /1* data::classification::oml::Dataset dataset; *1/ */

/*   /1* for ( size_t i=0; i<study.size_; i++) *1/ */
/*   /1* { *1/ */
/*   /1*   dataset = study.GetNext(); *1/ */
/*   /1*   PRINT(arma::size(dataset.inputs_)); *1/ */
/*   /1* } *1/ */

/*   data::oml::Dataset trainset,testset; */

/*   data::StratifiedSplit(dataset,trainset,testset,0.2); */
/*   PRINT_VAR(dataset.size_); */
/*   PRINT_VAR(dataset.dimension_); */
/*   PRINT_VAR(dataset.num_class_); */
/*   size_t repeat = 1000; */
/*   PRINT_VAR(trainset.size_); */
/*   PRINT_VAR(trainset.dimension_); */
/*   arma::irowvec Ns = arma::regspace<arma::irowvec>(10,1,100); */

/*   /1* arma::irowvec Ns = {1}; *1/ */
/*   /1* arma::irowvec Ns = arma::regspace<arma::irowvec>(10,1,10); *1/ */

/*   /1* algo::classification::LogisticRegression<> model(trainset.inputs_,trainset.labels_,3); *1/ */
/*   /1* mlpack::LogisticRegression<> model(trainset.inputs_.n_rows,0); *1/ */
/*   /1* model.Train(trainset.inputs_,trainset.labels_); *1/ */

/*   /1* PRINT_VAR(model.ComputeAccuracy(trainset.inputs_,trainset.labels_)); *1/ */

/*   /1* arma::irowvec Ns = arma::regspace<arma::irowvec>(2,10,100); *1/ */
/*   /1* src::LCurve<mlpack::LinearSVM<>,mlpack::Accuracy> lcurve(Ns,repeat,true,false,true); *1/ */
/*   /1* src::LCurve<algo::classification::KernelSVM<>,mlpack::Accuracy> lcurve(Ns,repeat,true,false,true); *1/ */
/*   /1* src::LCurve<algo::classification::KernelSVM<mlpack::GaussianKernel>, *1/ */
/*   /1* src::LCurve<algo::classification::LDC<>, *1/ */
/*   /1* src::LCurve<algo::classification::KernelSVM<mlpack::LinearKernel>, *1/ */
/*   /1* src::LCurve<algo::classification::QDC<>, *1/ */
/*               /1* mlpack::Accuracy> lcurve(Ns,repeat,true,false,true); *1/ */
/*   /1* src::LCurve<algo::classification::KernelSVM<mlpack::GaussianKernel>,mlpack::Accuracy, *1/ */
/*   /1* data::classification::Transformer<mlpack::data::StandardScaler,data::classification::oml::Dataset<>> trans(trainset); *1/ */
/*   /1* auto trainset_ = trans.Trans(trainset); *1/ */
/*   /1* auto testset_ = trans.Trans(testset); *1/ */
/*   /1* src::LCurve<algo::classification::OnevAll< *1/ */
/*   /1*   algo::classification::SVM<mlpack::GaussianKernel>>,mlpack::Accuracy, *1/ */
/*   /1* src::LCurve<algo::classification::OnevAll<mlpack::NaiveBayesClassifier<>>,mlpack::Accuracy, *1/ */
/*   /1*   data::N_StratSplit> lcurve(Ns,repeat,true,false,true); *1/ */
/*   /1* src::LCurve<algo::classification::QDC<>,mlpack::Accuracy> lcurve(Ns,repeat,true,true); *1/ */
/*   /1* src::LCurve<algo::classification::QDC<>,utils::BrierLoss> lcurve(Ns,repeat,true,true); *1/ */
/*   /1* src::LCurve<algo::classification::QDC<>,utils::AUC> lcurve(Ns,repeat,true,true); *1/ */
/*   /1* src::LCurve<algo::classification::SVM<mlpack::GaussianKernel,0>,mlpack::Accuracy> lcurve(Ns,repeat,true,true); *1/ */
/*   /1* src::LCurveHPT<algo::classification::SVM<mlpack::GaussianKernel,0>,mlpack::Accuracy> lcurve(Ns,repeat,0.2,true,true); *1/ */
/*   src::LCurveHPT<algo::classification::LDC<>,mlpack::Accuracy> lcurve(Ns,repeat,0.2,true,true); */
/*   arma::Mat<DTYPE> Cs = arma::logspace(1e-8,1e-2,5); */
/*   PRINT(arma::size(Cs)); */
/*   /1* src::LCurve<algo::classification::SVM<>,mlpack::Accuracy> lcurve(Ns,repeat,true,true); *1/ */
/*   /1* src::LCurve<algo::classification::NMC<>,mlpack::Accuracy> lcurve(Ns,repeat,true,true); *1/ */
/*   /1* src::LCurve<algo::classification::NNC<>,mlpack::Accuracy> lcurve(Ns,repeat,true,false,true); *1/ */
/*   /1* src::LCurve<algo::classification::LogisticRegression<>,mlpack::Accuracy> lcurve(Ns,repeat,true,true); *1/ */
/*   /1* src::LCurve<algo::classification::LogisticRegression<>,utils::CrossEntropy> lcurve(Ns,repeat,false,true); *1/ */
/*   /1* src::LCurve<algo::classification::LogisticRegression<>,utils::BrierLoss> lcurve(Ns,repeat,true,true); *1/ */
/*   /1* src::LCurve<algo::classification::OnevAll<mlpack::LogisticRegression<>>,mlpack::Accuracy> lcurve(Ns,repeat,false,true); *1/ */
/*   /1* src::LCurve<algo::classification::SVM<>,utils::CrossEntropy> lcurve(Ns,repeat,true,true); *1/ */
/*   /1* src::LCurve<algo::classification::SVM<mlpack::EpanechnikovKernel>,mlpack::Accuracy> lcurve(Ns,repeat,true,true); *1/ */
/*   /1* src::LCurve<algo::classification::SVM<mlpack::GaussianKernel>,mlpack::Accuracy> lcurve(Ns,repeat,true,true); *1/ */
/*   /1* src::LCurve<algo::classification::LDC<>,mlpack::Accuracy> lcurve(Ns,repeat,true,true); *1/ */
/*   /1* src::LCurve<algo::classification::LogisticRegression<>,mlpack::Accuracy> lcurve(Ns,repeat,false,true); *1/ */
/*   /1* src::LCurve<mlpack::RandomForest<>,utils::BrierLoss> lcurve(Ns,repeat,true,true); *1/ */
/*   /1* src::LCurve<mlpack::RandomForest<>,mlpack::Accuracy> lcurve(Ns,repeat,true,true); *1/ */
/*   /1* src::LCurve<algo::classification::SVM<>,utils::LogLoss> lcurve(Ns,repeat,true,false,true); *1/ */
/*   /1* src::LCurve<algo::classification::QDC<>,mlpack::Accuracy> lcurve(Ns,repeat,true,true); *1/ */
/*   /1* src::LCurve<algo::classification::LDC<>,utils::LogLoss> lcurve(Ns,repeat,false,false,true); *1/ */
/*   /1* src::LCurve<algo::classification::NNC<>,mlpack::Accuracy> lcurve(Ns,repeat,true,true); *1/ */
/*   /1* src::LCurve<mlpack::RandomForest<>,mlpack::Accuracy> lcurve(Ns,repeat,true,true); *1/ */
/*   /1* src::LCurve<algo::classification::NMC<>,mlpack::Accuracy> lcurve(Ns,repeat,true,false,true); *1/ */
/*   /1* src::LCurve<mlpack::AdaBoost<mlpack::ID3DecisionStump>,mlpack::Accuracy> lcurve(Ns,repeat,true,true); *1/ */
/*   /1* src::LCurve<mlpack::NaiveBayesClassifier<>,utils::CrossEntropy> lcurve(Ns,repeat,true,true); *1/ */
/*   /1* src::LCurve<mlpack::RandomForest<>,utils::AUC> lcurve(Ns,repeat,true,true); *1/ */
/*   /1* src::LCurve<algo::classification::NMC<>,mlpack::Accuracy> lcurve(Ns,repeat,false,true); *1/ */
/*   /1* src::LCurve<algo::classification::NMC<>,utils::AUC> lcurve(Ns,repeat,false,true); *1/ */
/*   /1* src::LCurve<mlpack::RandomForest<>,utils::CrossEntropy> lcurve(Ns,repeat,true,true); *1/ */
/*   /1* src::LCurve<mlpack::RandomForest<>,mlpack::Accuracy> lcurve(Ns,repeat,true,true); *1/ */
/*   lcurve.Split(trainset,testset,mlpack::Fixed(arma::unique(dataset.labels_).eval().n_elem),Cs); */
/*   /1* lcurve.Additive(trainset,testset,mlpack::Fixed(arma::unique(dataset.labels_).eval().n_elem),Cs); *1/ */
/*   /1* lcurve.RandomSet(dataset,arma::unique(dataset.labels_).eval().n_elem); *1/ */
/*   /1* lcurve.Additive(dataset,arma::unique(dataset.labels_).eval().n_elem); *1/ */
/*   PRINT_VAR(arma::size(lcurve.GetResults())); */
/*   /1* lcurve.Bootstrap(dataset.inputs_,dataset.labels_,arma::unique(dataset.labels_).eval().n_elem); *1/ */
/*   /1* lcurve.Split(trainset,testset,2,1.e-6); *1/ */
/*   /1* lcurve.Split(trainset,testset,arma::unique(dataset.labels_).eval().n_elem,1e-8); *1/ */
/*   /1* lcurve.Bootstrap(trainset.inputs_,trainset.labels_); *1/ */
/*   /1* src::LCurveHPT<algo::classification::LDC<>,mlpack::Accuracy> lcurve(Ns,repeat,0.2,false,true); *1/ */
/*   /1* auto lambdas = arma::logspace<arma::Row<DTYPE>>(-2,1,10); *1/ */
/*   /1* lcurve.Bootstrap(trainset.inputs_,trainset.labels_,mlpack::Fixed(arma::unique(dataset.labels_).eval().n_elem),lambdas); *1/ */
/*   /1* lcurve.Bootstrap(dataset.inputs_,dataset.labels_,arma::unique(dataset.labels_).eval().n_elem,1.); *1/ */
/*   /1* lcurve.Bootstrap(trainset.inputs_,trainset.labels_,size_t(3),0.1); *1/ */
/*   /1* lcurve.Bootstrap(trainset.inputs_,trainset.labels_,size_t(3),0.1); *1/ */
/*   /1* PRINT(arma::unique(dataset.labels_).eval().n_elem); *1/ */

/*   /1* src::LCurveHPT<algo::classification::SVM<>,mlpack::Accuracy> lcurve(Ns,repeat,0.2,true,false,true); *1/ */
/*   /1* lcurve.Split(trainset,testset,mlpack::Fixed(3),arma::logspace(-2,2,10)); *1/ */
/*   /1* lcurve.Split(trainset,testset,4,3); *1/ */
/*   /1* lcurve.Split(trainset_,testset_,1e-6); *1/ */
/*   /1* PRINT(lcurve.GetResults().has_nan()); *1/ */
/*   /1* lcurve.GetResults().save("fanSMO.csv",arma::csv_ascii); *1/ */

/*   PRINT_TIME(timer.toc()); */

  
/*   return 0; */
/* } */
