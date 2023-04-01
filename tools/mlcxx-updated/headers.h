/**
 * @file headers.h
 * @author Ozgur Taylan Turan
 *
 * header files for the whole mlcxx project
 *
 */

// Some easy definitions...
#define ARMA_WARN_LEVEL 1
#define PI
#define DOCTEST_CONFIG_IMPLEMENT
#define PRINT(...) std::cout << __VA_ARGS__ << std::endl; 
#define PRINT_VAR(...) std::cout << #__VA_ARGS__ << " := " << std::endl << __VA_ARGS__ << std::endl; 
#define PRINT_SEED(...) std::cout << " SEED               : " << __VA_ARGS__ << std::endl; 
#define PRINT_TIME(...) std::cout << " ELAPSED TIME (sec) : " << __VA_ARGS__ << std::endl; 

// standard
#include <list>
#include <cmath>
#include <string>
#include <cstdio>
#include <vector>
#include <variant>
#include <cstdlib>
#include <iomanip>
#include <numeric>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <filesystem>

// jem/jive
#include <jive/Array.h>
#include <jem/util/Timer.h>
#include <jem/base/String.h>
#include <jem/base/System.h>
#include <jem/util/Properties.h>


// boost 
#include <boost/any.hpp>
#include <boost/assert.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/throw_exception.hpp>
#include <boost/exception/diagnostic_information.hpp>
#include <boost/math/special_functions/factorials.hpp>
#include <boost/serialization/nvp.hpp>
//mlpack

#include <mlpack/core.hpp>
//#include <mlpack/methods/linear_regression.hpp>

// local
#include "utils/utils.h"
#include "algo/algo.h"
#include "opt/opt.h"
#include "src/src.h"
#include "models/models.h"
#include "stats/stats.h"
#include "stats/hypothesis.h"

// tests
#include "tests/doctest.h"
#include "tests/tests.h"
#include "exp/exp.h"
//#include "tests/test_arma.h"

// main functions
#include "main_input.h"
#include "main_func.h"


