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

//-----------------------------------------------------------------------------
//   main : our main function no need to do anything here...
//-----------------------------------------------------------------------------
int main ( int argc, char** argv )
{
  sims();

  return 0; 
}
