/**
 * @file main.cpp
 * @author Ozgur Taylan Turan
 *
 *
 */

#include <headers.h>

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
  jem::String kernel_type;

  props.get(model_type, "model.type");

  if (model_type == "LeastSquaresRegression")
  {
    model_ptr = new LeastSquaresRegression(modelProps, std::get<0>(datasets));
  }
  
  else if (model_type == "KernelLeastSquaresRegression")
  {
    props.get(kernel_type, "model.kernel.type");
    model_ptr = new KernelLeastSquaresRegression<mlpack::GaussianKernel>
                                            (modelProps, std::get<0>(datasets));
  }


  model_ptr -> DoObjective(objProps, std::get<0>(datasets));
  std::cout << " [ Objective Elapsed Time : "  << timer.toDouble() 
                                                       << " ]" << std::endl;

}

//-----------------------------------------------------------------------
//   main
//-----------------------------------------------------------------------

int main ( int argc, char** argv )
{
  doctest::Context context;

  // !!! THIS IS JUST AN EXAMPLE SHOWING HOW DEFAULTS/OVERRIDES ARE SET !!!

  // defaults
  //context.addFilter("test-case-exclude", "*math*"); 
  context.setOption("abort-after", 5);
  context.setOption("order-by", "file");

  context.applyCommandLine(argc, argv);

  // overrides
  context.setOption("no-breaks", true);

  int res = context.run(); 

  if(context.shouldExit()) 
      return res;          
  
  //// REGISTER YOUR PREDEFINED FUNCTIONS HERE
  typedef void (*FuncType)( );
  std::map<std::string, FuncType> m;
  m["test_smpkr"] = test_smpkr;
  m["test_combine"] = test_combine;
  m["test_functional"] = test_functional;
  m["test_lc"] = test_lc;
  m["genlrlc"] = genlrlc;
  m["read_data"] = read_data;
  m["gd_lm"] = gd_lm;
  m["eig_decay"] = eig_decay;
  m["cvm_2samp"] = cvm_2samp;
  jem::String log_file;

  jem::util::Timer timer;

  std::filesystem::path path(argv[1]); 

  //BOOST_ASSERT_MSG( argc == 2, "Need to provide a .pro input file or a function name!");

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
    mlpack::RandomSeed(seed);

    input_run(props);
    std::cout.put('\n');
    std::cout << "***[ Total Elapsed Time : "  << timer.toDouble() << " ]***";
    std::cout.put('\n');
  }
  else
  {

    BOOST_ASSERT_MSG((m.find(argv[1]) != m.end()), "REGISTER YOUR FUNCTION!!");
    m[argv[1]](); 
  }
  return 0; // the result from doctest is propagated here as well
}
