/**
 * @file main.cpp
 * @author Ozgur Taylan Turan
 *
 * TODO: Maybe make it easier to read by making names smaller etc with using
 *        statements!!!
 *
 * TODO: Seperate the functions in their respective files.
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
// mlpack
#include <mlpack/methods/linear_regression/linear_regression.hpp>
#include <mlpack/core/cv/simple_cv.hpp>
#include <mlpack/core/data/save.hpp>
#include <mlpack/core/hpt/hpt.hpp>
#include <mlpack/core/cv/k_fold_cv.hpp>
#include <mlpack/core/cv/metrics/mse.hpp>
// local
#include "utils/convert.h"
#include "utils/datagen.h"
#include "utils/covmat.h"
#include "utils/save.h"
#include "algo/kernelridge.h"
#include "algo/semiparamkernelridge.h"
#include "src/learning_curve.h"
#include "data/data_prep.h"
#include "model/model.h"
#include "model/leastsquares.h"

// standard
#include <variant>
#include <cstdio>
#include <vector>
#include <filesystem>

const char*  DATA_PROP        = "data";
const char*  MODEL_PROP       = "model";
const char*  OBJECTIVE_PROP   = "objective";

//-----------------------------------------------------------------------
//   input_run
//-----------------------------------------------------------------------



void input_run(jem::util::Properties props)
{

  jem::util::Properties dataProps = props.getProps(DATA_PROP);
  jem::util::Properties modelProps = props.getProps(MODEL_PROP);
  jem::util::Properties objProps = props.getProps(OBJECTIVE_PROP);

  jem::util::Timer timer;
  timer.start(); 
  auto datasets = DataPrepare(dataProps);
  std::cout << " [ Data Preparation Elapsed Time : "  << timer.toDouble() 
                                                       << " ]" << std::endl;
  std::cout.put('\n');

  timer.start(); 
  
  BaseModel* model_ptr;
  jem::String model_type;

  props.get(model_type, "model.type");

  if (model_type == "LeastSquaresRegression");
  {
    model_ptr = new LeastSquaresRegression(modelProps, std::get<0>(datasets));
  }

  model_ptr -> DoObjective(objProps, std::get<0>(datasets));
  std::cout << " [ Objective Elapsed Time : "  << timer.toDouble() 
                                                       << " ]" << std::endl;

}


//-----------------------------------------------------------------------
//   test
//-----------------------------------------------------------------------
#include <boost/preprocessor/repetition/repeat.hpp>

#define DECL(z, n, text) text ## n = n;

void test()
{
  int D, Ntrn, Ntst; D=1; Ntrn=5; Ntst=1000;
  double a, p, eps; a = 1.0; p = 0.; eps = 0.1;
  utils::data::regression::Dataset trainset(D, Ntrn, eps);
  utils::data::regression::Dataset testset(D, Ntst, eps);

  trainset.Generate(a, p, "Sine");
  testset.Generate(a, p, "Sine");
  auto inputs = trainset.inputs;
  auto labels = arma::conv_to<arma::rowvec>::from(trainset.labels);
  algo::regression::SemiParamKernelRidge<mlpack::kernel::GaussianKernel,
                                         utils::data::regression::SineGen> 
                      model(inputs,labels, 0., size_t(20), 1.);


  arma::rowvec pred_labels;
  model.Predict(inputs, pred_labels);
  
  

  //utils::data::regression::SineGen funcs(3);

  //arma::mat psi = funcs.Predict(inputs, "Phase");
  //std::cout << psi << std::endl;
  
  //arma::rowvec beta;
  //for(int i=0; i<10; i++)
  //{
  //  beta = arma::ones<arma::rowvec>(10)*i;
  //  std::cout << beta << std::endl;
  //}
  //  std::cout << beta << std::endl;
}

//-----------------------------------------------------------------------
//   main
//-----------------------------------------------------------------------

int main ( int argc, char** argv )
{
  // REGISTER YOUR PREDEFINED FUNCTIONS HERE
  typedef void (*FuncType)( );
  std::map<std::string, FuncType> m;
  m["test"] = test;

  jem::String log_file;

  jem::util::Timer timer;

  BOOST_ASSERT_MSG( argc == 2, "Need to provide a .pro input file or a function name!");

  std::filesystem::path path(argv[1]); 
  if ( path.extension() == ".pro" )
  {

    timer.start();
    jem::util::Properties props;
    props.parseFile(argv[1]);

    int seed = 24;  // KOBEE!!
    props.find(seed, "seed");

    std::cout.put('\n');
    std::cout << " [ Seed : "  << seed  << " ]" << std::endl;
    std::cout.put('\n');
    arma::arma_rng::set_seed(seed);
    mlpack::math::RandomSeed(seed);

    input_run(props);
    std::cout.put('\n');
    std::cout << "***[ Total Elapsed Time : "  << timer.toDouble() << " ]***";
    std::cout.put('\n');
  }
  else
  {

    BOOST_ASSERT_MSG((m.find(argv[1]) != m.end()), "Function Not Found!!");
    m[argv[1]](); 
  }

  return 0;
}
