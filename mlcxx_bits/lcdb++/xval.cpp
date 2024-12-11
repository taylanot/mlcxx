/**
 * @file xval.cpp
 * @author Ozgur Taylan Turan
 *
 * Let's check xval results
 */

#include <headers.h>



//-----------------------------------------------------------------------------
//   main : our main function no need to do anything here...
//-----------------------------------------------------------------------------
int main ( int argc, char** argv )
{
  std::filesystem::path dir = EXP_PATH/"11_12_24/k200/ldc";
  std::filesystem::create_directories(dir);


  using Dataset = data::classification::oml::Dataset<>;
  arma::irowvec ids = {11,37,39,53,61};

  for (size_t i=0; i<ids.n_elem; i++)
  {
    Dataset dataset(ids[i]);
    xval::KFoldCV<algo::classification::LDC<>, mlpack::Accuracy> xv(200, dataset.inputs_, dataset.labels_,true);
    xv.TrainAndEvaluate(dataset.num_class_,0.0).save(dir/(std::to_string(i)+".csv"),arma::csv_ascii);
  } 
}
