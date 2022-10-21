/**
 * @file register.h
 * @author Ozgur Taylan Turan
 *
 * Simple registration system for types that can be read from input files
 * for know only kernels and models are registered! For now only using partial
 * specialization to achive this dynamic type setting.
 *
 * TODO: Try to find a smarter way for doing this...
 * 
 *
 *
 */

#ifndef REGISTER_H
#define REGISTER_H

// standard
#include <map>
#include <string>

template <int N>
struct ModelRegister;

template <int N>
struct KernelRegister;

template<>
struct ModelRegister<1>
{
  typedef mlpack::regression::LinearRegression modelClass;
};

template<>
struct ModelRegister<2>
{
  typedef mlpack::regression::KernelRidge<mlpack::kernel::GaussianKernel> 
    modelClass;
};

int RegisterLookUp(std::string name)
{
  std::map<std::string, int> MODELMAP = {
      { "Linear",           1 },
      { "KernelRidge",      2 }
  };
  
  return MODELMAP[name];
}
#endif
