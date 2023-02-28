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

//#include "serialization.hpp"
//#include "test_catch_tools.hpp"
//#include "catch.hpp"

using namespace mlpack;

int main ( int argc, char** argv )
{
  arma::mat predictors;
  predictors = { {  0, 1, 2, 4, 8, 16 },
                 { 16, 8, 4, 2, 1,  0 } };
  arma::rowvec responses = "0 2 4 3 8 8";

  LinearRegression lr(predictors, responses);
  lr.ComputeError(predictors, responses);
  return 0;
}
