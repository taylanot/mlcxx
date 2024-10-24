/**
 * @file splitting.cpp
 * @author Ozgur Taylan Turan
 *
 * Trying to create splitting algo
 */

#include <headers.h>

// Main function
int main() {
  arma::mat X = {{0, 2}, {0, 0}, {2, 1}, {3, 4}, {4, 3}};
  arma::inplace_trans(X);
  PRINT(X);
  arma::Row<int> y = {-1, -1, -1, 1, 1};
  arma::mat X1, X2;
  arma::Row<int> y1,y2;


  /* data::Split(X,y,X1,X2,y1,y2,size_t(2)); */
  /* PRINT(X1) */
  /* PRINT(X2) */
  /* PRINT(y1) */
  /* PRINT(y2) */


  return 0;
}


