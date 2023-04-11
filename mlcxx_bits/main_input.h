/**
 * @file main_inputfile.h
 * @author Ozgur Taylan Turan
 *
 *
 */

#ifndef MAIN_INPUTFILE_H
#define MAIN_INPUTFILE_H

const char*  DATA_PROP        = "data";
const char*  MODEL_PROP       = "model";
const char*  OBJECTIVE_PROP   = "objective";

//-----------------------------------------------------------------------------
//   input_run
//-----------------------------------------------------------------------------

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
  else
  {
    // dummy initialization to avoid warning!
    model_ptr = new LeastSquaresRegression(modelProps, std::get<0>(datasets));
    throw std::runtime_error ( "Not implemented :)!" );
  }

  model_ptr -> DoObjective(objProps, std::get<0>(datasets));
  std::cout << " [ Objective Elapsed Time : "  << timer.toDouble() 
                                                       << " ]" << std::endl;

}

#endif
