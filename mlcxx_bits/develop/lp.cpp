/**
 * @file lp.cpp
 * @author Ozgur Taylan Turan
 *
 * Trying to create a linear programming routine using HiGHS
 */

#include <headers.h>

int main()
{

  arma::rowvec c = {-3, -4};

  arma::mat G = {{2, -1},
                 {1, 2},
                 {-1, 0},
                 {0, -1}};

  arma::rowvec h = {8,6,0,0};

  arma::mat A= {{1, 1}};
  arma::rowvec b= {1};
  arma::rowvec x0 ;
  
  opt::linprog(x0,c,G,h,A,b);

  return 0;
}
