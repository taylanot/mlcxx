/**
 * @file gd.h
 * @author Ozgur Taylan Turan
 *
 * Gradient descent problem with weird learning curve
 *
 */

#ifndef YAWLC_GD_H 
#define YAWLC_GD_H

namespace experiments {
namespace yawlc {

void gd ( )
{

  std::filesystem::path dir = "gd"; 

  utils::data::regression::Dataset dataset(conf::D, conf::N);
  dataset.Generate(conf::a, conf::b, conf::type);

 
  {

    auto id_arch = arma::regspace<arma::ivec>(1,3);
    auto id_nonlin = arma::regspace<arma::ivec>(1,4);

    auto it_arch = id_arch.begin();
    auto it_arch_end = id_arch.end();

    auto it_nonlin = id_nonlin.begin();
    auto it_nonlin_end = id_nonlin.end();
    
    for (; it_arch!=it_arch_end; it_arch++)
    {
      it_nonlin = id_nonlin.begin();
      for (; it_nonlin!=it_nonlin_end; it_nonlin++)
      {
        src::regression::LCurve<algo::regression::ANN<ens::GradientDescent>,
                                mlpack::MSE> lcurve_nn(conf::Ns,conf::repeat);

        std::filesystem::path filename = std::to_string(*it_arch)+
                                         "_"+std::to_string(*it_nonlin)+".csv";

        lcurve_nn.Generate(conf::dir_yawlc/dir/filename,
                           dataset.inputs_, dataset.labels_,
                           size_t(*it_arch), size_t(*it_nonlin));
      }
    }
  }
  //{ 
  //  auto lrs = arma::linspace<arma::vec>(0.0001,0.001,50);

  //  auto it = lrs.begin();
  //  auto it_end = lrs.end();

  //  size_t tag = 0; 

  //  std::filesystem::path dir = "linear-lr";

  //  for (; it!=it_end; it++)
  //  {

  //    src::regression::LCurve<algo::regression::ANN<ens::GradientDescent>,
  //                            mlpack::MSE> lcurve_nn(conf::Ns,conf::repeat);

  //    std::filesystem::path filename = std::to_string(tag)+".csv";

  //    lcurve_nn.Generate(conf::dir_yawlc/dir/filename,
  //                       dataset.inputs_, dataset.labels_,
  //                       conf::archtype, conf::nonlintype,
  //                       double(*it), size_t(100000), double(1e-5));
  //  }
  //}
}

void sgd ( )
{

  std::filesystem::path dir = "sgd"; 
 
  utils::data::regression::Dataset dataset(conf::D, conf::N);
  dataset.Generate(conf::a, conf::b, conf::type);

 
  //{

  //  auto id_arch = arma::regspace<arma::ivec>(1,3);
  //  auto id_nonlin = arma::regspace<arma::ivec>(1,4);

  //  auto it_arch = id_arch.begin();
  //  auto it_arch_end = id_arch.end();

  //  auto it_nonlin = id_nonlin.begin();
  //  auto it_nonlin_end = id_nonlin.end();
  //  
  //  for (; it_arch!=it_arch_end; it_arch++)
  //  {
  //    it_nonlin = id_nonlin.begin();
  //    for (; it_nonlin!=it_nonlin_end; it_nonlin++)
  //    {
  //      src::regression::LCurve<algo::regression::ANN<ens::StandardSGD>,
  //                              mlpack::MSE> lcurve_nn(conf::Ns,conf::repeat);

  //      std::filesystem::path filename = std::to_string(*it_arch)+
  //                                       "_"+std::to_string(*it_nonlin)+".csv";

  //      lcurve_nn.Generate(conf::dir_yawlc/dir/filename,
  //                         dataset.inputs_, dataset.labels_,
  //                         size_t(*it_arch), size_t(*it_nonlin));
  //    }
  //  }
  //}

  //{
  //  auto lrs = arma::linspace<arma::vec>(0.0001,0.01,100);

  //  auto it = lrs.begin();
  //  auto it_end = lrs.end();

  //  size_t id = 0;

  //  std::filesystem::path subdir = "lin_relu-lr";

  //  for (; it!=it_end; it++)
  //  {
  //    src::regression::LCurve<algo::regression::ANN<ens::StandardSGD>,
  //                            mlpack::MSE> lcurve_nn(conf::Ns,conf::repeat);

  //    std::filesystem::path filename = std::to_string(id)+".csv";

  //    lcurve_nn.Generate(conf::dir_yawlc/dir/subdir/filename,
  //                       dataset.inputs_, dataset.labels_,
  //                       size_t(2), size_t(1), double(*it), size_t(32));
  //    id++;
  //  }
  //}
  
  //  // First problematic learning rate
  //{
  //  double lr = 0.0029;

  //  std::filesystem::path subdir = "lin_relu-lr:0.0029";

  //    src::regression::LCurve<algo::regression::ANN<ens::StandardSGD>,
  //                            mlpack::MSE> lcurve_nn(conf::Ns,conf::repeat);

  //    std::filesystem::path filename = "lc.csv";

  //    lcurve_nn.Generate(true, conf::dir_yawlc/dir/subdir/filename,
  //                       dataset.inputs_, dataset.labels_,
  //                       size_t(1), size_t(1), double(lr), size_t(32));
  //}
  //// Problematic one 
  
   ////arma::mat x = {-4.3905368941219143e+00,-4.6724124732473618e+00,-1.5345211907181451e+00,-3.1389110821398494e+00,1.1890080246351218e+00,-4.9826622497104145e+00,-2.8273987887578671e+00,3.4802743774781657e+00,4.9886235377012174e+00,-4.9828981053068180e+00,-4.4523791585049137e+00,4.5159796199905209e+00,2.8295937690856112e+00,1.3928466599355769e+00,-4.8712717864621968e+00,2.4934629568197932e+00,3.2260623554516030e+00,4.8880606668814295e+00,9.6484293092575335e-01,4.1434136456683301e+00,5.2006012719291306e-01,4.7948946287051353e+00,-2.5094172609045144e+00,-4.7111405709496719e+00,-4.1592101919374000e+00,-2.7153692792426014e+00,4.8703115309386540e+00,4.1593269193510913e+00,-3.3699469678303209e+00,-2.7711019143918287e+00};
   //arma::mat x = {-1.5345211907181451e+00,-3.1389110821398494e+00,1.1890080246351218e+00,-4.9826622497104145e+00,-2.8273987887578671e+00,3.4802743774781657e+00,4.9886235377012174e+00,-4.9828981053068180e+00,-4.4523791585049137e+00,4.5159796199905209e+00,2.8295937690856112e+00,1.3928466599355769e+00,-4.8712717864621968e+00,2.4934629568197932e+00,3.2260623554516030e+00,4.8880606668814295e+00,9.6484293092575335e-01,4.1434136456683301e+00,5.2006012719291306e-01,4.7948946287051353e+00,-2.5094172609045144e+00,-4.7111405709496719e+00,-4.1592101919374000e+00,-2.7153692792426014e+00,4.8703115309386540e+00,4.1593269193510913e+00,-3.3699469678303209e+00,-2.7711019143918287e+00};

   //auto lrs = arma::linspace<arma::vec>(0.0001,0.01,100);

   //auto it = lrs.begin();
   //auto it_end = lrs.end();

   //for (; it!=it_end; it++)
   //{
   //  algo::regression::ANN<ens::StandardSGD> model(x,x,size_t(1),size_t(1), double(0.0029), size_t(32));
   //  PRINT(model.ComputeError(x,x));
   //}

  //}
  
  {
    for (size_t i=0;i<100;i++)
    {
     PRINT("*********************************************************");
     arma::mat x = arma::randn(1,30);
     PRINT_VAR(x)
     algo::regression::ANN<ens::StandardSGD> model(x,x,size_t(2),size_t(1), double(0.0081), size_t(32),10000000, 1e-6, true);
     //model.Parameters() = {1,-1};
     //model.Train(x,x);
     //algo::regression::ANN<ens::StandardSGD> model2(x,x,size_t(2),size_t(1), double(0.0001), size_t(32),10000000, 1e-6, true);
     ////model2.Parameters() = {1,-1};
     ////model2.Train(x,x);
     //algo::regression::ANN<ens::StandardSGD> model3(x,x,size_t(2),size_t(1), double(1e-5), size_t(32),1000000, 1e-6, true);
     //model3.Parameters() = {1,-1};
     //model3.Train(x,x);
     PRINT_VAR(model.Parameters());
     //PRINT_VAR(model2.Parameters());
     //PRINT_VAR(model3.Parameters());
    }
  }
}



} // ywalc namespace
} // experiments

#endif
