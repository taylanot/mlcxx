/**
 * @file mlcxx.cpp
 * @author Ozgur Taylan Turan
 *
 * Main file of mlcxx where you do not have to do anything...
 */

#include <headers.h>

const int  SEED = 8 ; // KOBEEEE

//-----------------------------------------------------------------------------
//   main : our main function no need to do anything here...
//-----------------------------------------------------------------------------

int main ( int argc, char** argv )
{
  //{
  //  utils::data::classification::Dataset dataset(2,10000,2);
  //  dataset.Generate("Hard");
  //  //dataset.Save("dataset.csv");

  //  arma::irowvec Ns = arma::regspace<arma::irowvec>(2,10,1000);

  //  src::classification::LCurve<algo::classification::NMC,
  //                                mlpack::Accuracy> lcurve(Ns,100);

  //  lcurve.StratifiedGenerate(dataset.inputs_, dataset.labels_ );
  //  lcurve.Save("stratified.csv");

  //  lcurve.Generate(dataset.inputs_, dataset.labels_ );
  //  lcurve.Save("notstratified.csv");
  //}
  //{
  //  arma::irowvec Ns = arma::regspace<arma::irowvec>(1,1,100);
  //  arma::rowvec lambdas = arma::linspace<arma::rowvec>(1e-5,1.,20);
  //  #pragma omp parallel for 
  //  for(size_t i=0; i<20; i++)
  //  {
  //    utils::data::classification::Dataset dataset;
  //    dataset.Load("llc_checks/elinear/"+std::to_string(i)+".bin");
  //    src::regression::LCurve<mlpack::LinearRegression,
  //                                 mlpack::Accuracy> lcurve(Ns,size_t(1e2));
  //    lcurve.Generate(dataset.inputs_,
  //                    arma::conv_to<arma::rowvec>::from(dataset.labels_),
  //                    size_t(lambdas(i)));
  //    lcurve.Save("llc_checks/lc/"+std::to_string(i)+".csv");
  //  }
  //}

  {

    arma::rowvec lambdas = arma::linspace<arma::rowvec>(1e-5,1.,20);
    arma::irowvec Ns = arma::regspace<arma::irowvec>(2,10,1000);
    src::regression::LCurve<mlpack::LinearRegression,
                                  mlpack::Accuracy> lcurve(Ns,size_t(1e2));

    arma::mat mean_test_error(100,400);
    arma::field<std::string> header(400);

    for ( size_t data_id=0; data_id<20; data_id++ )
    {
      utils::data::regression::Dataset dataset;
      std::filesystem::path  dataset_file = "llc_checks/elinear";
      dataset_file = dataset_file/(std::to_string(data_id)+".bin");

      dataset.Load(dataset_file.string(),1,1,true,false);
      arma::rowvec labels = arma::conv_to<arma::rowvec>::from
                                                            (dataset.labels_);

      for ( size_t model_id=0; model_id<20; model_id++ )
      {
        size_t counter = 20*data_id+model_id;

        std::filesystem::path model_path;
        model_path = std::to_string(model_id);

        header(counter) = model_path.string();

        lcurve.Generate(dataset.inputs_,labels,
                        double(lambdas(model_id)));

        mean_test_error.col(counter) = std::get<1>(lcurve.stats_).row(0).t();

    }

    std::filesystem::path dir = "llc_checks/lc";
    std::filesystem::create_directories(dir);
    header.save((dir/"headers.csv").string());
    mean_test_error.save(arma::csv_name((dir/"test_mean2.csv").string(), header)); 
    }
  }

  return 0; 
}
