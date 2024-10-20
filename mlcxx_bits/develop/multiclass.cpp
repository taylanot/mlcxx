/**
 * @file multiclass.cpp
 * @author Ozgur Taylan Turan
 *
 * Trying to create a multi-class classifier 
 */

#include <headers.h>

int main() 
{
  data::classification::oml::Dataset dataset(11);

  /* algo::classification::OnevAll<mlpack::LogisticRegression<arma::mat>> */ 
  algo::classification::OnevAll<algo::classification::SVM<>>
    model(dataset.inputs_,dataset.labels_,arma::unique(dataset.labels_).eval().n_elem,2);
  /* algo::classification::MultiClass<algo::classification::KernelSVM<>> */ 
  /*   model(dataset.inputs_,dataset.labels_,1.,0.01); */
  arma::Row<size_t> preds;
  arma::Mat<DTYPE> probs;
  model.Classify(dataset.inputs_,preds,probs);
  PRINT_VAR(arma::size(probs));
  PRINT_VAR(arma::size(preds));
  PRINT(model.ComputeAccuracy(dataset.inputs_,dataset.labels_));

  /* arma::irowvec Ns = arma::regspace<arma::irowvec>(1,1,140); */
  /* size_t rep = 100; */
  /* src::LCurve<algo::MultiClass<mlpack::LogisticRegression<arma::mat>>, */
  /*             mlpack::Accuracy, */
  /*             utils::StratifiedSplit> LC(Ns,rep); */
  /* LC.Bootstrap(dataset.inputs_, dataset.labels_); */
  /* LC.test_errors_.save("check.csv", arma::csv_ascii); */
 

  
  return 0;
}
