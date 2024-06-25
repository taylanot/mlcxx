/**
 * @file lin.h
 * @author Ozgur Taylan Turan
 *
 * Linear Model Regression absurd repetition. 
 *
 */

#ifndef LCI_lin_H 
#define LCI_lin_H

namespace experiments {
namespace lci {

void classify ( )
{

  utils::data::classification::Dataset dataset(2, conf::N, 2);
  dataset.Generate(conf::classtype);

  src::classification::LCurve<algo::classification::LDC<>,
                              mlpack::Accuracy> lcurve(conf::Ns,conf::repeat);

  lcurve.Generate(dataset.inputs_,dataset.labels_,1e-5);

  arma::mat test_errors_ = lcurve.test_errors_;
  arma::mat Ns = arma::conv_to<arma::mat>::from(conf::Ns);
  arma::mat results = arma::join_vert(Ns, test_errors_);
  utils::Save(conf::dir_lci/"Banana.csv", results);

}

void lin2 ( )
{

    arma::field<std::string> header;
    arma::mat B;
    B.load(arma::csv_name(conf::dataset_dir/conf::filename, header));
    arma::inplace_trans(B);
    utils::data::regression::Dataset dataset;
    dataset.inputs_ = B.row(0);
    dataset.labels_ = B.row(1);

    src::regression::LCurve<mlpack::LinearRegression,
                            mlpack::MSE> lcurve(conf::Ns,conf::repeat);
    lcurve.Generate(dataset.inputs_,dataset.labels_);

    arma::mat test_errors_ = lcurve.test_errors_;

    arma::mat Ns = arma::conv_to<arma::mat>::from(conf::Ns);
    arma::mat results = arma::join_vert(Ns, test_errors_);

    utils::Save(conf::dir_lci/conf::filename, results);

}
void lin3 ( )
{

    arma::field<std::string> header;
    arma::mat B;
    B.load(arma::csv_name(conf::dataset_dir/conf::filename, header));
    arma::inplace_trans(B);
    utils::data::regression::Dataset dataset;
    utils::data::regression::Dataset trainset;
    utils::data::regression::Dataset testset;

    dataset.inputs_ = B.row(0);
    dataset.labels_ = B.row(1);
    dataset.size_ = B.n_cols;
    dataset.dimension_ = 1;

    utils::data::Split(dataset,trainset,testset,double(0.5));

    src::regression::LCurve<mlpack::LinearRegression,
                            mlpack::MSE> lcurve(conf::Ns,conf::repeat);
    lcurve.ParallelGenerate(trainset,testset);

    arma::mat test_errors_ = lcurve.test_errors_;

    arma::mat Ns = arma::conv_to<arma::mat>::from(conf::Ns);
    arma::mat results = arma::join_vert(Ns, test_errors_);
    utils::Save(conf::dir_lci/"seperate_test.csv", results);
    
    //lcurve.Save(conf::dir_lci/conf::filename_var);

    //src::regression::LCurve<mlpack::LinearRegression,
    //                        mlpack::MSE> lcurve2(conf::Ns,conf::repeat);
    //lcurve2.Generate(dataset.inputs_,arma::conv_to<arma::rowvec>::from(dataset.labels_));
    //lcurve2.Save(conf::dir_lci/conf::filename);

}


void lin4 ( )
{

    arma::field<std::string> header;
    arma::mat B;
    B.load(arma::csv_name(conf::dataset_dir/conf::filename, header));
    arma::inplace_trans(B);
    utils::data::regression::Dataset dataset;
    dataset.inputs_ = B.row(0);
    dataset.labels_ = B.row(1);
    src::regression::VariableLCurve<mlpack::LinearRegression,
                            mlpack::MSE> lcurve(conf::Ns,conf::repeats);
    lcurve.Generate(dataset.inputs_,arma::conv_to<arma::rowvec>::from(dataset.labels_));

    arma::mat test_errors_ = lcurve.test_errors_;

    arma::mat Ns = arma::conv_to<arma::mat>::from(conf::Ns);
    arma::mat results = arma::join_vert(Ns, test_errors_);
    utils::Save(conf::dir_lci/conf::filename_var, results);
    
    //lcurve.Save(conf::dir_lci/conf::filename_var);

    //src::regression::LCurve<mlpack::LinearRegression,
    //                        mlpack::MSE> lcurve2(conf::Ns,conf::repeat);
    //lcurve2.Generate(dataset.inputs_,arma::conv_to<arma::rowvec>::from(dataset.labels_));
    //lcurve2.Save(conf::dir_lci/conf::filename);

}

void lin( )
{
  std::filesystem::path dir_dist = "gaussian"; 


  utils::data::regression::Dataset dataset(conf::D, conf::N);

  auto it = conf::stds.begin();
  auto it_end = conf::stds.end();

  for (;it != it_end; it++)
  {
    dataset.Generate(1.,0.,conf::type,double(*it));

    std::filesystem::path dir_noise = "noise-"+std::to_string(int(*it));
    arma::mat error(conf::repeat,conf::Ns.n_elem);

    PRINT("RUNNNING-lin_custom...");

    #pragma omp parallel for collapse(2)
    for (size_t i=0; i < size_t(conf::Ns.n_elem) ; i++)
    {
      for(size_t j=0; j < size_t(conf::repeat); j++)
      {
        const auto data = utils::data::Split(dataset.inputs_,
                                            dataset.labels_,
                                            size_t(conf::Ns(i)));
        arma::mat X1 = std::get<0>(data); 
        arma::mat Ytrn = std::get<2>(data); 
        arma::mat Xtrn = arma::join_vert(X1, arma::ones<arma::rowvec>(X1.n_cols));

        arma::mat X2 = std::get<1>(data); 
        arma::mat Ytst = std::get<3>(data); 
        arma::mat Xtst = arma::join_vert(X2, arma::ones<arma::rowvec>(X2.n_cols));

        arma::mat W = arma::pinv(Xtrn*Xtrn.t())*Xtrn*Ytrn.t();
        arma::rowvec res = (W.t()*Xtst-Ytst);

        error (j,i) = arma::dot(res,res) / double(Xtst.n_cols) ;
      }
    }
    std::filesystem::create_directories(conf::dir_lci/dir_dist/dir_noise);
    utils::Save(conf::dir_lci/dir_dist/dir_noise/"error.csv",error);
  }
    
}

//void lin2( )
//{
//  std::filesystem::path dir_dist = "exp-sin"; 
//
//
//  utils::data::regression::Dataset dataset(conf::D, conf::N);
//
//  auto it = conf::stds.begin();
//  auto it_end = conf::stds.end();
//
//  for (;it != it_end; it++)
//  {
//    dataset.Generate(1.,0.,conf::type,double(*it));
//
//    std::filesystem::path dir_noise = "noise-"+std::to_string(int(*it));
//    arma::mat error(conf::repeat,conf::Ns.n_elem);
//
//    PRINT("RUNNNING-lin_custom...");
//
//    #pragma omp parallel for collapse(2)
//    for (size_t i=0; i < size_t(conf::Ns.n_elem) ; i++)
//    {
//      for(size_t j=0; j < size_t(conf::repeat); j++)
//      {
//        const auto data = utils::data::Split(dataset.inputs_,
//                                            dataset.labels_,
//                                            conf::Ns(i));
//        arma::mat X1 = std::get<0>(data); 
//        arma::mat Ytrn = std::get<2>(data); 
//        arma::mat Xtrn = arma::join_vert(X1, arma::ones<arma::rowvec>(X1.n_cols));
//
//        arma::mat X2 = std::get<1>(data); 
//        arma::mat Ytst = std::get<3>(data); 
//        arma::mat Xtst = arma::join_vert(X2, arma::ones<arma::rowvec>(X2.n_cols));
//
//        arma::mat W = arma::pinv(Xtrn*Xtrn.t())*Xtrn*Ytrn.t();
//        arma::rowvec res = (W.t()*Xtst-Ytst);
//
//        error (j,i) = arma::dot(res,res) / double(Xtst.n_cols) ;
//      }
//    }
//    std::filesystem::create_directories(conf::dir_lci/dir_dist/dir_noise);
//    utils::Save(conf::dir_lci/dir_dist/dir_noise/"error.csv",error);
//  }
//    
//}
} // lci namespace
} // experiments

#endif
