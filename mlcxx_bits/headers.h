/**
 * @file headers.h
 * @author Ozgur Taylan Turan
 *
 * header files for the whole mlcxx project
 *
 */

// default seed
const unsigned int SEED = 8 ; // KOBEEEE
// Some easy definitions...
//#define ARMA_USE_HDF5
#define ARMA_WARN_LEVEL 1
#define ARMA_USE_BLAS
/* #define ARMA_USE_LAPACK */
#define DOCTEST_CONFIG_IMPLEMENT
#define PRINT(...) std::cout << '\n' <<  __VA_ARGS__ << '\n' << std::endl; 
#define PRINT_INFO(...) std::cout << '\n' <<  "[INFO]:"  << __VA_ARGS__  << '\n' << std::endl; 
#define PRINT_GOOD() std::cout << '\n' <<  "[INFO]: ALL WENT GOOD!" << '\n' << std::endl; 
#define PRINT_BAD()  std::cout << '\n' <<  "[INFO]: OEPS STH IS WRONG!!!" << '\n' << std::endl;
#define PRINT_VAR(...) std::cout << '\n' << #__VA_ARGS__ << " := " << std::endl << __VA_ARGS__ << '\n' << std::endl; 
#define PRINT_SEED(...) std::cout << '\n' << "[INFO]: SEED               : " << __VA_ARGS__ << '\n' << std::endl; 
#define PRINT_TIME(...) std::cout << '\n' << "[INFO]: ELAPSED TIME (sec) : " << __VA_ARGS__ << '\n' << std::endl; 
#define PRINT_MLPACK_VERSION() std::cout << '\n' << "[INFO]: MLPACK_VERSION : " << mlpack::util::GetVersion() << '\n' << std::endl; 
#define PRINT_ARMADILLO_VERSION() std::cout << '\n' << "[INFO]: ARMADi_VERSION : " << arma::arma_version::as_string() << '\n' << std::endl; 

#define PRINT_ERR(...) std::cerr << '\n' <<  __VA_ARGS__ << '\n' << std::endl; 

#ifdef DISABLE_WARNINGS
#define WARN(msg) // Do nothing
#else
#define WARN(msg) std::clog << "\n" << "Warning: " << msg << std::endl << "\n"
#endif

#ifdef DISABLE_LOGS
#define LOG(msg) // Do nothing
#else
#define LOG(msg) std::clog  << "\n" << "Log  : " << msg << std::endl << "\n"
#endif

#ifdef DISABLE_ERRORS
#define ERR(msg) // Do nothing
#else
#define ERR(msg) std::cerr <<  "\n" << "Error  : " << msg << std::endl << "\n" 
#endif

//#define PRINT_REQ() std::cout << '\n' << "[INFO]: MAIN_PACKAGES: " << "armadillo " <<arma::arma_version::as_string() << " / " << mlpack::util::GetVersion() << " / ensmallen " << ens::version::as_string() << "/ boost " << BOOST_LIB_VERSION << " / jem " << jem::JEM_VERSION <<'\n' << std::endl; 
#define PRINT_REQ() std::cout << '\n' << "[INFO]: MAIN_PACKAGES: " << "armadillo " <<arma::arma_version::as_string() << " / " << mlpack::util::GetVersion() << " / ensmallen " << ens::version::as_string() << "/ boost " << BOOST_LIB_VERSION << '\n' << std::endl; 

// Define the precision for what is to come...
#ifndef DTYPE
#define DTYPE double
#endif



// standard
#include <any>
#include <list>
#include <cmath>
#include <string>
#include <cstdio>
#include <chrono>
#include <vector>
#include <variant>
#include <cstdlib>
#include <iomanip>
#include <numeric>
#include <typeinfo>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <filesystem>
#include <limits>
#include <random>
#include <cassert>
#include <regex>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <unistd.h>
#include <csignal>


// Define some of the constants paths
const std::filesystem::path MLCXX_PATH = std::filesystem::current_path();
const std::filesystem::path DATASET_PATH = MLCXX_PATH.parent_path()/"datasets";
const std::filesystem::path EXP_PATH = MLCXX_PATH.parent_path()/".exp";

// boost 
#include <boost/assert.hpp>
/* #include <boost/program_options.hpp> */
#include <boost/version.hpp>
#include <boost/throw_exception.hpp>
#include <boost/exception/diagnostic_information.hpp>
#include <boost/math/special_functions/factorials.hpp>
#include <boost/math/special_functions/binomial.hpp>
#include <boost/math/special_functions/beta.hpp>
#include <boost/math/distributions/empirical_cumulative_distribution_function.hpp>
#include <boost/math/statistics/t_test.hpp>

#include <boost/serialization/nvp.hpp>
#include <boost/random.hpp>

// mlpack
#include <mlpack.hpp>
// HiGHS Linear Optimization package
#include <highs/Highs.h>
// curl  
#include <curl/curl.h>
/* #include <mlpack/methods/linear_svm.hpp> */

// local
#include "utils/utils.h"
#include "data/data.h"
#include "xval/xval.h"

#include "opt/opt.h"
#include "stats/stats.h"
#include "algo/algo.h"
#include "src/src.h"



