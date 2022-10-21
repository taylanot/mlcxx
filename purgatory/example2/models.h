#ifndef MODELS_H
#define MODELS_H

#include <torch/torch.h>

//struct LinearModel : torch::nn::Module 
//{
//  LinearModel(int64_t N, int64_t M);
//
//  virtual torch::Tensor forward(torch::Tensor input);
//
//  torch::Tensor W, b;
//};

struct LinearModelImpl: torch::nn::Module 
{
  LinearModelImpl(int N, int M);
  
  virtual torch::Tensor forward(torch::Tensor x);

  torch::nn::Linear lin1;
};
#endif
