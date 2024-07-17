/**
 * @file compare.cpp
 * @author Ozgur Taylan Turan
 *
 * Checking some peaking curves
 */

//#define ARMA_DONT_USE_LAPACK
//#define ARMA_DONT_USE_BLAS
//#define ARMA_DONT_USE_ARPACK
//#define ARMA_DONT_USE_OPENMP
#include <headers.h>



//-----------------------------------------------------------------------------
//   main : our main function no need to do anything here...
//-----------------------------------------------------------------------------
/* int main ( int argc, char** argv ) */
/* { */
/*   arma::wall_clock timer; */
/*   timer.tic(); */


/*   std::filesystem::path path = ".16_10_23/wiggle/lin2_tanh_hpt"; */
/*   std::filesystem::create_directories(path); */
/*   std::filesystem::current_path(path); */

/*   utils::data::regression::Dataset trainset(1, 100000); */
/*   utils::data::regression::Dataset testset(1, 10000); */

/*   arma::irowvec Ns = arma::regspace<arma::irowvec>(10,1,12); */

/*   src::LCurveHPT<algo::regression::ANN<ens::StandardSGD>, mlpack::MSE> */ 
/*                                                     LC(Ns,1,0.2,false,false,true); */
/*   /1* src::LCurve<algo::regression::ANN<ens::StandardSGD>, mlpack::MSE> *1/ */ 
/*   /1*                                                   LC(Ns,1,false,false,true); *1/ */

/*   trainset.Generate("Linear",0.0); */
/*   testset.Generate("Linear",0.0); */

/*   arma::Row<size_t> layer_info = {5,5,1}; */
/*   size_t arch = 1; */
/*   arma::Row<size_t> bss = {10,20,32}; */
/*   arma::Row<double> lrs = {0.001,0.01,0.1}; */
/*   bool early = true; */
/*   size_t nonlin = 3; */


/*   if (std::atoi(argv[1]) == 0) */
/*   { */
/*     LC.Additive(trainset.inputs_,trainset.GetLabels(0), */
/*                 mlpack::Fixed(arch),mlpack::Fixed(nonlin), */
/*                 mlpack::Fixed(early), bss, lrs ); */
/*     LC.test_errors_.save("additive.csv", arma::csv_ascii); */
/*   } */
/*   else if (std::atoi(argv[1]) == 1) */
/*   { */
/*     LC.Split(trainset,testset, */
/*              mlpack::Fixed(layer_info),mlpack::Fixed(nonlin), */
/*              mlpack::Fixed(early), bss, lrs ); */
/*     LC.test_errors_.save("split.csv", arma::csv_ascii); */
/*   } */
/*   else if (std::atoi(argv[1]) == 2) */
/*   { */
/*     LC.Bootstrap(trainset.inputs_,trainset.GetLabels(0), */
/*              mlpack::Fixed(layer_info),mlpack::Fixed(nonlin), */
/*              mlpack::Fixed(early), bss, lrs ); */
/*     LC.test_errors_.save("bootstrap.csv", arma::csv_ascii); */
/*   } */

/*   PRINT_TIME(timer.toc()); */

/*   return 0; */
/* } */

int main ( int argc, char** argv )
{
  arma::wall_clock timer;
  timer.tic();


  std::filesystem::path path = ".16_10_23/wiggle/adjusted";
  std::filesystem::create_directories(path);
  std::filesystem::current_path(path);
  utils::data::regression::Dataset trainset(1, 100000);
  utils::data::regression::Dataset testset(1, 10000);

  arma::irowvec Ns = arma::regspace<arma::irowvec>(1,1,100);

  src::LCurve<algo::regression::ANN<ens::StandardSGD>, mlpack::MSE> 
                                                    LC(Ns,100,false,false,true);

  trainset.Generate("Linear",0.0);
  testset.Generate("Linear",0.0);

  arma::Row<size_t> layer_info = {5,5,1};
  size_t arch_type = 1;
  bool early = true;
  size_t nonlin = 3;


  if (std::atoi(argv[1]) == 0)
  {
    LC.Additive(trainset.inputs_,trainset.GetLabels(0),layer_info,3,early);
    LC.test_errors_.save("additive.csv", arma::csv_ascii);
  }

  else if (std::atoi(argv[1]) == 1)
  {
    LC.Split(trainset,testset,layer_info,3,early);
    LC.test_errors_.save("split.csv", arma::csv_ascii);
  }

  else if (std::atoi(argv[1]) == 2)
  {
    LC.Bootstrap(trainset.inputs_,trainset.GetLabels(0),layer_info,3,early);
    LC.test_errors_.save("bootstrap.csv", arma::csv_ascii);
  }

  PRINT_TIME(timer.toc());

  return 0;
}
