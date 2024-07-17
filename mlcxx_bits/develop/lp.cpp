/**
 * @file lp.cpp
 * @author Ozgur Taylan Turan
 *
 * Trying to create a linear programming routine using HiGHS
 */

#define DTYPE float 
#include <headers.h>
int main()
{

  /* arma::rowvec c = {-3, -4}; */

  /* arma::mat G = {{2, -1}, */
  /*                {1, 2}, */
  /*                {-1, 0}, */
  /*                {0, -1}}; */

  /* arma::rowvec h = {8,6,0,0}; */

  /* arma::mat A= {{1, 1}}; */
  /* arma::rowvec b= {1}; */
  /* arma::rowvec x0 ; */
  /* arma::mat G_; */
  /* arma::rowvec h_; */

  /* /1* opt::linprog(x0,c,G,h,A,b); *1/ */
  /* opt::linprog(x0,c,G_,h_,A,b,true,true); */
  /* PRINT(x0); */

  arma::frowvec c = {-3, -4};

  arma::fmat G = {{2, -1},
                 {1, 2},
                 {-1, 0},
                 {0, -1}};

  arma::frowvec h = {8,6,0,0};

  arma::fmat A= {{1, 1}};
  arma::frowvec b= {1};
  arma::frowvec x0 ;
  arma::fmat G_;
  arma::frowvec h_;

  /* opt::linprog(x0,c,G,h,A,b); */
  opt::linprog(x0,c,G_,h_,A,b,true,true);
  PRINT(x0);


  return 0;
}
