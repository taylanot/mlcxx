#include <jem/base/String.h>
#include <jem/base/System.h>
#include <jem/util/Timer.h>
#include <jem/util/Properties.h>
#include <torch/torch.h>
#include <iostream>
#include <typeinfo>
#include <map>
#include <string>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

#include "models.h"
//int main () {
//  LinearModel model(1,1);
//  std::cout << model.forward(torch::ones({100,1})) << std::endl;
//  return 0;
//};
using jem::System;
using namespace jem::util;
////-----------------------------------------------------------------------
////   run
////-----------------------------------------------------------------------
//
//
////// Register Parameter
////struct LinearModel: torch::nn::Module {
////  LinearModel(int64_t N, int64_t M) {
////    W = register_parameter("W", torch::randn({N, M}));
////    b = register_parameter("b", torch::randn(M));
////  }
////  torch::Tensor forward(torch::Tensor input) {
////  return torch::addmm(b, input, W);
////  }
////  torch::Tensor W, b;
////};
////
////// Register Submodule
////struct LinearModel2: torch::nn::Module{
////  LinearModel2(int64_t N, int64_t M): 
////    linear(register_module("linear", torch::nn::Linear(N,M))) {
////    another_bias = register_parameter("b", torch::randn(M));
////    }
////  torch::Tensor forward(torch::Tensor input) {
////    return linear(input); //+ another_bias;
////  }
////  torch::nn::Linear linear;
////  torch::Tensor another_bias;
////
////  torch::Tensor fit(torch::Tensor x){
////      return x;
////  }
////};
//
////struct NetImpl: torch::nn::Module {       // replaced Net by NetImpl
////  NetImpl(int N, int M)                                // replaced Net by NetImpl
////      : lin1(torch::nn::LinearOptions(N,M).bias(false))
////  {
////    register_module("lin1", lin1);
////  }
////
////  torch::Tensor forward(torch::Tensor x) {
////    x = lin1->forward(x);
////    return x;
////  }
////  torch::nn::Linear lin1;
////};
//
//
TORCH_MODULE(LinearModel);

void train(LinearModel model, torch::Tensor data, int epochs) {
  auto optimizer = torch::optim::SGD(model->parameters(), 0.01);
  for (int epoch=1; epoch <= epochs; epoch++){
    auto out = model->forward(data);
    optimizer.zero_grad();
    auto loss = torch::mse_loss(out, torch::ones({1,1})*2);
    loss.backward();
    optimizer.step();
    //std::cout << loss << std::endl;
  };
};

int run ()
{
  //using jem::String;
  //using jem::io::endl;
  //using jem::util::Properties;

  //Properties  conf;
  //String      header;
  //std::map<String, int> dict;
  //dict["ALI"] = 1;
  //for (auto element : dict){
  //  System::out() << element.first << jem::io::endl;
  //};
  //int         maxIter;
  //double      initValue;
  //double      tolerance;

  //conf.parseFile ( "input.pro" );

  //conf.find ( header,    "model" );
  //conf.get  ( maxIter,   "solver.maxIter",   1, 1000 );
  //conf.get  ( initValue, "solver.initValue" );
  //conf.get  ( tolerance, "solver.tolerance", 1e-20, 1e20 );
  //
  //System::out() << header << endl;
  //torch::manual_seed(0);
  //torch::Tensor tensor = torch::rand({2, 3});
  //std::cout << tensor << std::endl;

  // using namespace boost::numeric::ublas;
  // matrix<double> m (3, 3);
  // for (unsigned i = 0; i < m.size1 (); ++ i)
  //   for (unsigned j = 0; j < m.size2 (); ++ j)
  //     m (i, j) = 3 * i + j;
  // std::cout << m << std::endl;
  //torch::manual_seed(8); 
  //LinearModel model(4,5);
  //for (const auto& p: model.parameters()) {
  //  std::cout << p << std::endl;
  //}
  //std::cout << model.forward(torch::ones({2,4})) << std::endl;

  //torch::manual_seed(8); 
  //LinearModel2 model2(4,5);
  //std::cout << model2.fit(torch::ones({2,4})) << std::endl;
  LinearModel model(1,1);
  //std::cout << typeid(model).name() << std::endl;
  //std::cout << model->forward(torch::ones({2,4})) << std::endl;
  torch::Tensor  data = torch::ones({1,1});
  //for (const auto& p: model->parameters()) {
  //  std::cout << p << std::endl;
  //};
  
  Timer timer;
  timer.start();
  for(int i=0; i < 100; i++){
  train(model,data, 1000);
  };
  std::cout << timer.toDouble() << std::endl;
  for (const auto& p: model->parameters()) {
    std::cout << p << std::endl;
  };

  // A a;
  // std::cout << a.a << std::endl;

  return 0;
}

int main ()
{
  System::exec ( run );
}
