/**
 * @file normass.cpp
 * @author Ozgur Taylan Turan
 *
 * Let's see if the error distribution change effects somethings?
 */

#include <headers.h>



//-----------------------------------------------------------------------------
//   main : our main function no need to do anything here...
//-----------------------------------------------------------------------------
int main ( int argc, char** argv )
{
  arma::wall_clock timer;
  timer.tic();
  
  std::filesystem::path dir = EXP_PATH/"06_12_24";
  std::filesystem::create_directories(dir);

  data::regression::Dataset trainset(3, 5000);
  data::regression::Dataset testset(3, 10000);


  arma::irowvec Ns = arma::regspace<arma::irowvec>(1,100,4500);

  src::LCurve<mlpack::LinearRegression<>,mlpack::MSE> LC(Ns,10000,true,true);


  {
    trainset.Generate(std::string("Linear"),1.);
    testset.Generate(std::string("Linear"),1.);

    LC.Split(trainset,testset,0,false);

    LC.GetResults().save(dir/"norm1.csv", arma::csv_ascii);
  }

  {
    stats::Sampler< boost::random::beta_distribution<> > beta(0.5,0.5);
    trainset.Generate(std::string("Linear"),0.);
    testset.Generate(std::string("Linear"),0.);
    trainset.labels_ += beta.Sample(1,trainset.labels_.n_elem);
    testset.labels_ += beta.Sample(1,testset.labels_.n_elem);
    LC.Split(trainset,testset,0,false);
    LC.GetResults().save(dir/"beta0505.csv", arma::csv_ascii);
  }

  {
    stats::Sampler< boost::random::beta_distribution<> > beta(1,3);
    trainset.Generate(std::string("Linear"),0.);
    testset.Generate(std::string("Linear"),0.);
    trainset.labels_ += beta.Sample(1,trainset.labels_.n_elem);
    testset.labels_ += beta.Sample(1,testset.labels_.n_elem);
    LC.Split(trainset,testset,0,false);
    LC.GetResults().save(dir/"beta13.csv", arma::csv_ascii);
  }

  {
    stats::Sampler< boost::random::chi_squared_distribution<> > chi(2);
    trainset.Generate(std::string("Linear"),0.);
    testset.Generate(std::string("Linear"),0.);
    trainset.labels_ += chi.Sample(1,trainset.labels_.n_elem);
    testset.labels_ += chi.Sample(1,testset.labels_.n_elem);
    LC.Split(trainset,testset,0,false);
    LC.GetResults().save(dir/"chi2.csv", arma::csv_ascii);
  }

  {
    stats::Sampler< boost::random::chi_squared_distribution<> > chi(5);
    trainset.Generate(std::string("Linear"),0.);
    testset.Generate(std::string("Linear"),0.);
    trainset.labels_ += chi.Sample(1,trainset.labels_.n_elem);
    testset.labels_ += chi.Sample(1,testset.labels_.n_elem);
    LC.Split(trainset,testset,0,false);
    LC.GetResults().save(dir/"chi5.csv", arma::csv_ascii);
  }

  /* { */
  /*   trainset.Generate(std::string("Sine"),0.); */
  /*   testset.Generate(std::string("Sine"),0.); */

  /*   LC.Split(trainset, testset,0,false); */

  /*   LC.test_errors_.save(dir/"peak-sine0.csv", arma::csv_ascii); */
  /* } */

  /* { */
  /*   trainset.Generate(std::string("Linear"),1.); */
  /*   testset.Generate(std::string("Linear"),1.); */

  /*   LC.Split(trainset, testset,0,false); */

  /*   LC.test_errors_.save(dir/"peak-linear1.csv", arma::csv_ascii); */
  /* } */

  /* { */
  /*   trainset.Generate(std::string("Sine"),1.); */
  /*   testset.Generate(std::string("Sine"),1.); */

  /*   LC.Split(trainset, testset,0,false); */

  /*   LC.test_errors_.save(dir/"peak-sine1.csv", arma::csv_ascii); */
  /* } */

  /* PRINT_TIME(timer.toc()); */

  /* return 0; */
}



