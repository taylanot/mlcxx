/**
 * @file test.cpp
 * @author Ozgur Taylan Turan
 *
 * Compiling the tests
 */

#define DISABLE_LOGS
#define DISABLE_WARNINGS

#include <lcpp.h>

#include "tests/tests.h"



//-----------------------------------------------------------------------------
//   main : our main function no need to do anything here...
//-----------------------------------------------------------------------------
int main ( int argc, char** argv )
{
  mlpack::RandomSeed(SEED);

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
 
  PRINT_SEED(SEED);

  PRINT_TIME(timer.toc());

  DEPEND_INFO();
  
  return 0; 
}
