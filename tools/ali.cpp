/**
 * @file tests/linear_regression_test.cpp
 *
 * Test for linear regression.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/linear_regression.hpp>
#include <iostream>

//#include "serialization.hpp"
//#include "test_catch_tools.hpp"
//#include "catch.hpp"
using namespace mlpack;
using namespace arma;
using namespace std;

int main ()
{
  RandomSeed(22);
  //arma_rng::set_seed(22);
  mat matrix;
  matrix = randn(2,3);
  cout << matrix << endl;
  ///arma::mat matrix;
  ///mlpack::data::DatasetInfo info;
  ///mlpack::data::Load("mlcxx/iris.csv", matrix, info, true, true);
  ///std::cout << matrix.t() << std::endl;
  return 0;
}
