/**
 * @file svm.cpp
 * @author Ozgur Taylan Turan
 *
 * Trying to create kernelized svm
 */

#include <headers.h>
// Define the RBF kernel function

// Main function
int main() {
  arma::mat X = {{0, 2}, {0, 0}, {2, 1}, {3, 4}, {4, 3}};
  arma::inplace_trans(X);
  arma::Row<int> y = {-1, -1, -1, 1, 1};
 
  double C = 1.0;

  // Create SVM object and train
  algo::classification::KernelSVM svm(C);
  svm.Train(X, y);
  arma::Row<int> pred;
  svm.Classify(X, pred);
  PRINT(pred)

  return 0;
}


