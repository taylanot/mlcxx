/**
 * @file mean.cpp
 * @author Ozgur Taylan Turan
 *
 * This file is for making main mean estimation of various distributions!
 */

#include <headers.h>

void simulate ( arma::mat full, double real_mean, std::filesystem::path dist)
{
  std::filesystem::path path;

  path = "build/risk_estim/outputs"/dist/std::to_string(full.n_cols);
  std::filesystem::create_directories(path);

  arma::mat data, e, c, l, t;
  e.resize(full.n_rows,1);
  c.resize(full.n_rows,1);
  l.resize(full.n_rows,1);
  t.resize(full.n_rows,1);
  for (size_t i=0; i<full.n_rows; i++)
  {
    data = full.row(i);
    e(i,0) = arma::mean(data,1).eval()(0,0)-real_mean;
    t(i,0) = utils::tmean(data).eval()(0,0)-real_mean;
    c(i,0) = utils::catoni(data).eval()(0,0)-real_mean;
    l(i,0) = utils::lee(data).eval()(0,0)-real_mean;
  } 
  e.save(path/"e.bin",arma::raw_binary);
  t.save(path/"t.bin",arma::raw_binary);
  c.save(path/"c.bin",arma::raw_binary);
  l.save(path/"l.bin",arma::raw_binary);
}

void sims ( )
{
  size_t rep = 1000000;
  for (size_t k=5; k<11; k++)
  {
    auto pareto = utils::dists::pareto(rep,k);
    double pareto_m = 1;

    auto lognorm = utils::dists::lognorm(rep,k);
    double lognorm_m = std::exp(0.5);

    auto bimodal = utils::dists::gaussbimodal(rep,k);
    double bimodal_m = 0;

    auto norm = arma::randn(rep,k);
    double norm_m = 0;

    simulate( pareto, pareto_m, "pareto" ); 
    simulate( lognorm, lognorm_m, "lognorm" ); 
    simulate( bimodal, bimodal_m, "bimodal" ); 
    simulate( norm, norm_m, "norm" ); 

  }

}

void gens()
{
  std::filesystem::path path = "build/risk_estim/outputs";
  utils::data::regression::Dataset trainset(1, 100);
  utils::data::regression::Dataset testset(1, 1000);
  trainset.Generate("Outlier-1");
  testset.Generate("Outlier-1");


  arma::irowvec Ns = arma::regspace<arma::irowvec>(2,1,20);

  size_t rep = 1000;

  src::regression::LCurve<mlpack::LinearRegression,mlpack::MSE> lcurve(Ns,rep);
  src::regression::LCurve<mlpack::LinearRegression,mlpack::MSE> lcurve2(Ns,rep);

  lcurve.ParallelGenerate(trainset, testset, 0., true);
  PRINT_VAR(arma::size(lcurve.test_errors_))
  /* lcurve2.ParallelGenerate(trainset, testset, 10., false); */

  /* lcurve.test_errors_.save(path/"test_0.bin",arma::raw_binary); */
  /* lcurve2.test_errors_.save(path/"test_10.bin",arma::raw_binary); */

}

void look()
{
  std::filesystem::path path = "build/risk_estim/outputs";
  arma::mat test0, test10;

  test0.load(path/"test_0.bin",arma::raw_binary);
  test10.load(path/"test_10.bin",arma::raw_binary);
  test0 = arma::reshape(test0,1000000,19).t();
  test10 = arma::reshape(test10,1000000,19).t();

  test0.shed_col(0);test10.shed_col(0);
  size_t rep  = 1000000;
  size_t k = 5;
  arma::uvec idx;
  arma::mat data0, data10;

  arma::mat e0(test0.n_rows,rep);
  arma::mat t0(test0.n_rows,rep);
  arma::mat l0(test0.n_rows,rep);
  arma::mat c0(test0.n_rows,rep);

  arma::mat e10(test10.n_rows, rep);
  arma::mat t10(test10.n_rows, rep);
  arma::mat l10(test10.n_rows, rep);
  arma::mat c10(test10.n_rows, rep);

  double ecount = 0 ;
  double ccount = 0 ;
  double tcount = 0 ;
  double lcount = 0 ;
  for (size_t j=0; j<rep; j++)
  {
    idx = arma::randperm(rep).head(k);
    data0 = test0.cols(idx);
    data10 = test10.cols(idx);

    for (size_t i=0; i<10; i++)
    {
      arma::mat d0 = data0.row(i);
      arma::mat d10= data10.row(i);

      e0(i,j) = arma::mean(d0,1).eval()(0,0);
      t0(i,j) = utils::tmean(d0).eval()(0,0);
      c0(i,j) = utils::catoni(d0).eval()(0,0);
      /* l0(i,j) = utils::lee(d0).eval()(0,0); */
   
      e10(i,j) = arma::mean(d10,1).eval()(0,0);
      t10(i,j) = utils::tmean(d10).eval()(0,0);
      c10(i,j) = utils::catoni(d10).eval()(0,0);
      /* l10(i,j) = utils::lee(d10).eval()(0,0); */
      if (e10(i,j) > e0(i,j) )
        ecount++;
      if (c10(i,j) > c0(i,j) )
        ccount++;
      if (t10(i,j) > t0(i,j) )
        tcount++;
    }
  }
  std::cout << "e:" << ecount / rep * 100 << std::endl;
  std::cout << "t:" << tcount / rep * 100 << std::endl;
  std::cout << "c:" << ccount / rep * 100 << std::endl;
  /* std::cout << "l" << arma::find(l0<l10).eval().n_elem / rep * 100 << std::endl; */
}

//-----------------------------------------------------------------------------
//   main : our main function no need to do anything here...
//-----------------------------------------------------------------------------
int main ( int argc, char** argv )
{

  /* sims(); */
  /* gens(); */
  look();

  return 0; 
}
