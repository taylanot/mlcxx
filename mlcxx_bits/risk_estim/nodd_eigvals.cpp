/**
 * @file nodd_eigvals.cpp
 * @author Ozgur Taylan Turan
 *
 * This file is for showing double descent is related to regularization 
 * caused by precision.
 *
 */
#define DTYPE double

#include <headers.h>


int main ( int argc, char** argv )
{
  /* mlpack::RandomSeed(SEED); */

  std::filesystem::path path = EXP_PATH/"13_08_23";
  std::filesystem::create_directories(path);
  std::filesystem::current_path(path);

  arma::wall_clock timer;
  timer.tic();

  /* arma::irowvec Ds = {5,10,20}; */
  arma::irowvec Ds = {5};
  /* DTYPE stddev = .1; */

  size_t rep = 1000000;
  /* for (size_t i=0;i<Ds.n_elem;i++) */
  /* { */
  /*   arma::irowvec Ns = arma::regspace<arma::irowvec>(2,1,Ds[i]+20); */
  /*   std::filesystem::create_directories(std::to_string(Ds[i])+"/eigs"); */
  /*   std::filesystem::create_directories(std::to_string(Ds[i])+"/rconds"); */
  /*   for (size_t k=0;k<Ns.n_elem;k++) */
  /*   { */
  /*     arma::Mat<DTYPE> eigs(Ds[i],rep); */
  /*     arma::Row<DTYPE> rconds(rep); */
  /*     for (size_t j=0;j<rep;j++) */
  /*     { */
  /*       data::regression::Dataset dataset(Ds(i),Ns[k]); */
  /*       dataset.Generate(std::string("Linear"),stddev); */
  /*       const arma::Mat<DTYPE> A = dataset.inputs_ * dataset.inputs_.t(); */
  /*       arma::Mat<DTYPE> U,V; */
  /*       arma::Col<DTYPE> S; */
  /*       arma::svd_econ(U,S,V,A); */
  /*       eigs.col(j) = S; */
  /*       rconds.col(j) = arma::rcond(A); */
  /*     } */
  /*     eigs.save(std::to_string(Ds[i])+"/eigs/"+std::to_string(Ns[k])+".csv",arma::csv_ascii); */
  /*     rconds.save(std::to_string(Ds[i])+"/rconds/"+std::to_string(Ns[k])+".csv",arma::csv_ascii); */
  /*   } */
  /* } */
  
  for (size_t i=0;i<Ds.n_elem;i++)
  {
    arma::irowvec Ns = arma::regspace<arma::irowvec>(10,1,11);
    for (size_t k=0;k<Ns.n_elem;k++)
    {
      arma::Row<DTYPE> max(rep);
      arma::Row<DTYPE> min(rep);
      arma::Row<DTYPE> val(rep);
      auto mean = arma::zeros<arma::Col<DTYPE>>(Ds[i]);
      auto cov = arma::eye<arma::Mat<DTYPE>>(Ds[i],Ds[i]);
      for (size_t j=0;j<rep;j++)
      {
        arma::Mat<DTYPE> X = arma::mvnrnd(mean, cov, Ns[k]);
        arma::Mat<DTYPE> A = X*X.t();
        arma::Mat<DTYPE> U,V; 
        arma::Col<DTYPE> S;
        arma::svd_econ(U,S,V,A);
        arma::Col<DTYPE> s2(arma::size(S));

        int counter=0;
        for(size_t i=0; i < S.n_elem; ++i)
        {
          s2[i] = (S[i] != 0) ? DTYPE(DTYPE(1) / (S[i]*S[i])) : DTYPE(0);
          counter++;
        }
        max(j) = arma::max(S);
        min(j) = arma::min(S[arma::rank(A)-1]);
        val(j) = arma::accu(s2);
      }

      std::cout << "N :  "  << Ns[k] << "\n" << std::endl;

      PRINT_VAR(arma::mean(min))
      PRINT_VAR(arma::mean(max))
      PRINT_VAR(arma::mean(val))
    }
  }

  PRINT_TIME(timer.toc());

  return 0;
}
