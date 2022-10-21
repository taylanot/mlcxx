#include <torch/torch.h>
//#include <string>

#include "models.h"

// Date constructor
//Date::Date(int year, int month, int day)
//{
//    SetDate(year, month, day);
//}
//
//// Date member function
//void Date::SetDate(int year, int month, int day)
//{
//    m_month = month;
//    m_day = day;
//    m_year = year;
//}
//LinearModel::LinearModel(int64_t N, int64_t M) 
//{
//  W = register_parameter("W", torch::randn({N, M}));
//  b = register_parameter("b", torch::randn(M));
//}
//torch::Tensor LinearModel::forward(torch::Tensor input) 
//{
//  return torch::addmm(b, input, W);
//}
//

struct NetImpl: torch::nn::Module {       
  NetImpl(int N, int M)                  
      : lin1(torch::nn::LinearOptions(N,M).bias(false))
  {
    register_module("lin1", lin1);
  }

  torch::Tensor forward(torch::Tensor x) {
    x = lin1->forward(x);
    return x;
  }
  torch::nn::Linear lin1;
};

LinearModelImpl::LinearModelImpl(int N, int M) :
  lin1(torch::nn::LinearOptions(N,M).bias(false))
{
  register_module("lin1", lin1);
}
torch::Tensor LinearModelImpl::forward(torch::Tensor x) 
{
  return lin1->forward(x);
}
//struct LinearModelImpl: torch::nn::Module {
//  LinearModelImpl(int N, int M, string bias)
//      : _lin1(torch::nn::LinearOptions(N,M).bias(bias))
//  {
//    register_module("_lin1", _lin1);
//  }
//
//  torch::Tensor forward(torch::Tensor x) {
//    return _lin1->forward(x);
//    
//  }
//  torch::nn::Linear _lin1;
//};
//
//TORCH_MODULE(LinearModel);
