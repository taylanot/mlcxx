/**
 * @file mlcxx.cpp
 * @author Ozgur Taylan Turan
 *
 * Main file of mlcxx where you do not have to do anything...
 */

//#define ARMA_DONT_USE_LAPACK
//#define ARMA_DONT_USE_BLAS
//#define ARMA_DONT_USE_ARPACK
//#define ARMA_DONT_USE_OPENMP
#include <headers.h>

//-----------------------------------------------------------------------------
//   main : our main function no need to do anything here...
//-----------------------------------------------------------------------------
int main ( int argc, char** argv )
{

  mlpack::RandomSeed(SEED);

  arma::wall_clock timer;
  timer.tic();

  // Running the functions defined in funcs.cpp
  if ( argc > 1 ) 
  {
    std::filesystem::path path(argv[1]); 

    if ( strcmp(argv[1], "-no-run") != 0 )
    {
      PRINT("***Running a pre-specified routine in funcs.cpp...");
      func_run(argv[1]);
    }
    else
    {
      std::cout.put('\n');
      PRINT("***Only running main function, which is empty!***");
      std::cout.put('\n');
    }
  }

  PRINT_SEED(SEED);

  PRINT_GOOD(); 

  PRINT_TIME(timer.toc());
  
  return 0; 
}
