/**
 * @file main_func.h
 * @author Ozgur Taylan Turan
 *
 *
 */

#ifndef MAIN_FUNC_H
#define MAIN_FUNC_H

//-----------------------------------------------------------------------------
//   func_run
//-----------------------------------------------------------------------------

void func_run( char* argv )
{
  //// REGISTER YOUR PREDEFINED FUNCTIONS HERE
  typedef void (*FuncType)( );
  std::map<std::string, FuncType> m;

  //m["test_smpkr"] = test_smpkr;
  //m["test_combine"] = test_combine;
  //m["test_functional"] = test_functional;
  //m["test_lc"] = test_lc;
  //m["genlrlc"] = genlrlc;
  //m["read_data"] = read_data;
  //m["gd_lm"] = gd_lm;
  //m["eig_decay"] = eig_decay;
  //m["cvm_2samp"] = cvm_2samp;

  BOOST_ASSERT_MSG((m.find(argv) != m.end()),
   "You should register your function to 'main_func.h'!!");
  m[argv]();

}

#endif

