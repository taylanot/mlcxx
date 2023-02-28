/**
 * @file headers.h
 * @author Ozgur Taylan Turan
 *
 * header files for the whole mlcxx project
 *
 */

#define ARMA_WARN_LEVEL 1
// jem/jive
#include <jem/base/String.h>
#include <jem/util/Timer.h>
#include <jem/util/Properties.h>
#include <jem/base/System.h>
#include <jive/Array.h>
// boost 
#include <boost/assert.hpp>
#include <boost/any.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
#include <boost/throw_exception.hpp>
#include <boost/exception/diagnostic_information.hpp>
// mlpack
#include <mlpack/methods/linear_regression/linear_regression.hpp>
#include <mlpack/core/cv/simple_cv.hpp>
#include <mlpack/core/data/save.hpp>
#include <mlpack/core/data/load.hpp>
//#include <mlpack/core/data/dataset_mapper.hpp>
#include <mlpack/core/hpt/hpt.hpp>
#include <mlpack/core/cv/k_fold_cv.hpp>
#include <mlpack/core/cv/metrics/mse.hpp>
// local
#include "utils/system.h"
#include "utils/save.h"
#include "utils/load.h"
#include "utils/convert.h"
#include "utils/datagen.h"
#include "utils/covmat.h"
#include "algo/kernelridge.h"
#include "algo/semiparamkernelridge.h"
#include "utils/functionalpca.h"
#include "utils/functionalsmoothing.h"
#include "src/learning_curve.h"
#include "data/data_prep.h"
#include "model/model.h"
#include "model/leastsquaresregression.h"
#include "tests/test_arma.h"
//#include "tests/test_torch.h"
//#include "projects/tests.h"

// standard
#include <cstdio>
#include <vector>
#include <variant>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <filesystem>
