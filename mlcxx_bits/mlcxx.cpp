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

const int  SEED = 8 ; // KOBEEEE

//-----------------------------------------------------------------------------
//   main : our main function no need to do anything here...
//-----------------------------------------------------------------------------
int main ( int argc, char** argv )
{

  int seed = SEED;

  mlpack::RandomSeed(seed);

  arma::wall_clock timer;
  timer.tic();

  // Unit Testing Framework
  
  doctest::Context context;

  context.setOption("abort-after", 5);
  context.setOption("order-by", "file");

  context.applyCommandLine(argc, argv);

  context.setOption("no-breaks", true);

  int res = context.run(); 

  if(context.shouldExit()) 
      return res;          
 
  // Running the functions defined in funcs.cpp
  if ( argc > 1 ) 
  {
    std::filesystem::path path(argv[1]); 

    mlpack::RandomSeed(seed);

    if ( strcmp(argv[1], "-no-run") != 0 )
    {
      PRINT("***Running a function...");
      func_run(argv[1]);
    }
    else
    {
      std::cout.put('\n');
      PRINT("***Only running main function, which can be empty!***");
      std::cout.put('\n');
    }
  }

  PRINT_SEED(SEED);

  PRINT_GOOD(); 

  PRINT_TIME(timer.toc());
  
  return 0; 
}
