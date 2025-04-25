/**
 * @file new.cpp
 * @author Ozgur Taylan Turan
 *
 * I am creating a new learning curve class for better usabaility and cleaner
 * code...
 */

#include <headers.h>

// regression set
/* using DATASET = data::oml::Dataset<DTYPE>; */
/* using MODEL = mlpack::LinearRegression<>; */
/* using LOSS = mlpack::MSE; */
/* using SAMPLE = data::RandomSelect2; */

// classification set
using DATASET = data::oml::Dataset<size_t>;
using MODEL = mlpack::LogisticRegression<>;
using LOSS = mlpack::Accuracy;
using SAMPLE = data::RandomSelect;

int main (int argc, char** argv ) 
{
  auto& conf = CLIStore::getInstance();

  conf.Register<bool>("load",true);

  conf.Parse(argc,argv);

  DATASET data(4);
  PRINT_VAR(data.size_);
  auto lambdas = arma::linspace<arma::Row<DTYPE>>(0,1,10);
  auto Ns = arma::regspace<arma::Row<size_t>>(10,1,data.size_-1);

  if (!conf.Get<bool>("load"))
  { 
    lcurve::LCurve<MODEL,
                   DATASET,
                   SAMPLE,
                   LOSS> curve(data,Ns,size_t(10000),true,true);

    curve.Generate(0.2,lambdas);

  }  
  else
  {
    auto loaded = lcurve::LCurve<MODEL,DATASET,SAMPLE,LOSS>::Load
                            (std::string("LCurve.bin"));
    loaded->CheckStatus();
    PRINT_VAR(arma::size(arma::find_nonfinite(loaded->GetResults())));
    auto lambdas = arma::linspace<arma::Row<DTYPE>>(0,1,10);
    loaded->Generate(0.2,lambdas);
  }


  /* auto Ns = arma::regspace<arma::Row<size_t>>(10,1,11); */
  /* lcurve::LCurve<MODEL, */
  /*        DATASET, */
  /*        SAMPLE, */
  /*        /1* data::Bootstrap, *1/ */
  /*        /1* data::Additive, *1/ */
  /*        /1* mlpack::MSE> curve(Ns,size_t(2),0.2,true,true); *1/ */
  /*        LOSS> curve(data,Ns,size_t(1000000),true,true); */
  /* auto lambdas = arma::linspace<arma::Row<DTYPE>>(0,1,10); */
  /* curve.Generate( ); */
  /* PRINT_VAR(arma::size(arma::find_nonfinite(curve.GetResults()))); */

  // For looking at continueation 
  /* curve.test_errors_(0,0) = arma::datum::nan; */

  /* curve.test_errors_(2,1) = arma::datum::nan; */
  /* PRINT_VAR(curve.GetResults()); */
  /* curve.Continue(lambdas); */
  /* PRINT_VAR(curve.GetResults()); */
  /* curve.Continue(lambdas); */
  /* PRINT_VAR(curve.GetResults()); */


  /* arma::mat A(5, 5, arma::fill::randu); */

  /* A.fill(arma::datum::nan); */
  /*  arma::uvec indices = arma::find_nan(A); */
  /* PRINT(A); */
  /* PRINT_VAR(indices) */

  // Continue 
  
  /* auto loaded = lcurve::LCurve<MODEL,DATASET,SAMPLE,LOSS>::Load */
  /*                         (std::string("LCurve.bin")); */
  /* /1* auto task = std::move(*loaded); *1/ */
  /* /1* PRINT(task.GetResults()); *1/ */
  /* loaded->CheckStatus(); */
  /* PRINT_VAR(arma::size(arma::find_nonfinite(loaded->GetResults()))); */
  /* /1* PRINT(loaded->GetResults()); *1/ */
  /* auto lambdas = arma::linspace<arma::Row<DTYPE>>(0,1,10); */
  /* /1* alarm(1); *1/ */
  /* /1* loaded->Generate(0.2,lambdas); *1/ */
  /* loaded->Generate(); */

  /* PRINT(task.GetResults()); */
  /* PRINT_VAR(arma::size(arma::find_nan(task.GetResults()))); */
  /* task.Save("LCurve.bin"); */

  // Load Dataset
  /* Dataset data(212); */
  /* PRINT_VAR(data.size_); */
  /* data.Save("dataset.bin"); */
  /* auto loaded = Dataset::Load(std::string("dataset.bin")); */
  /* Dataset data2; */
  /* data2 = std::move(*loaded); */
  /* PRINT_VAR(arma::size(data2.inputs_)); */
  /* PRINT_VAR(arma::size(data2.labels_)); */




  return 0;
}

/* /1* using MODEL = algo::classification::LDC<>; *1/ */
/* using MODEL = mlpack::RandomForest<>; */
/* using LOSS = mlpack::Accuracy; */
/* using OPT = ens::GridSearch; */
/* int main ( ) */ 
/* { */
/*   Dataset data(451); */
  
/*   auto Ns = arma::regspace<arma::Row<size_t>>(10,1,15); */
/*   lcurve::LCurve<algo::classification::LDC<>, */
/*          Dataset, */
/*          data::RandomSelect, */
/*          /1* data::Bootstrap, *1/ */
/*          /1* data::Additive, *1/ */
/*          /1* mlpack::MSE> curve(Ns,size_t(2),0.2,true,true); *1/ */
/*          /1* mlpack::Accuracy> curve(Ns,size_t(10000),DTYPE(0.2),true,true); *1/ */
/*          /1* mlpack::Accuracy> curve(Ns,size_t(10000),false,true); *1/ */
/*          mlpack::Accuracy> curve(Ns,size_t(10000),double(0.2),false,true); */
/*   auto lambdas = arma::linspace<arma::Row<DTYPE>>(0,1,10); */
/*   curve.Generate(data,lambdas); */
/*   /1* alarm(1); *1/ */
/*   /1* curve.Generate(data,lambdas); *1/ */
/*   /1* PRINT_VAR(curve.GetResults()); *1/ */

/*   /1* // Hyperparameter tuning *1/ */ 
/*   /1* mlpack::HyperParameterTuner<MODEL,LOSS,mlpack::SimpleCV,OPT,arma::Mat<DTYPE>> tune(0.2, data.inputs_, data.labels_,size_t(2)); *1/ */
/*   /1* auto ls = arma::regspace<arma::Row<DTYPE>>(1,10,100); *1/ */
/*   /1* auto res = tune.Optimize(ls); *1/ */

/*   /1* /2* MODEL model = std::move(tune.BestModel()); *2/ *1/ */
/*   /1* /2* model.Train(data.inputs_,data.labels_,2,std::move(res)); *2/ *1/ */
/*   /1* // Unpack res and pass as arguments to Train *1/ */
  
/*   /1* /2* std::apply([&](auto&&... args) *2/ *1/ */
/*   /1* /2* { *2/ *1/ */
/*   /1* /2*   model.Train(data.inputs_, data.labels_, 2, std::forward<decltype(args)>(args)...); *2/ *1/ */
/*   /1* /2* }, res); *2/ *1/ */


/*   /1* MODEL model = std::apply([&](auto&&... args) *1/ */ 
/*   /1* { *1/ */
/*   /1*   return MODEL(data.inputs_,data.labels_,2, *1/ */
/*   /1*                std::forward<decltype(args)>(args)...); *1/ */
/*   /1* }, res); *1/ */
  

/*   /1* /2* PRINT_VAR(model.num_class_); *2/ *1/ */
/*   /1* /2* PRINT_VAR(model.lambda_); *2/ *1/ */
/*   /1* LOSS loss; *1/ */
/*   /1* PRINT_VAR(loss.Evaluate(model,data.inputs_,data.labels_)); *1/ */
/*   /1* /2* PRINT_VAR(model.ComputeAccuracy(data.inputs_,data.labels_)); *2/ *1/ */




/*   // For looking at continueation */ 
/*   /1* curve.test_errors_(0,0) = arma::datum::nan; *1/ */
/*   /1* curve.test_errors_(2,1) = arma::datum::nan; *1/ */
/*   /1* PRINT_VAR(curve.GetResults()); *1/ */
/*   /1* curve.Continue(lambdas); *1/ */
/*   /1* PRINT_VAR(curve.GetResults()); *1/ */
/*   /1* curve.Continue(lambdas); *1/ */
/*   /1* PRINT_VAR(curve.GetResults()); *1/ */


/*   /1* arma::mat A(5, 5, arma::fill::randu); *1/ */

/*   /1* A.fill(arma::datum::nan); *1/ */
/*   /1*  arma::uvec indices = arma::find_nan(A); *1/ */
/*   /1* PRINT(A); *1/ */
/*   /1* PRINT_VAR(indices) *1/ */

/*   // Continue */ 
  
/*   /1* auto loaded = LCurve_::Load(std::string("LCurve.bin")); *1/ */
/*   /1* LCurve_ task = std::move(*loaded); *1/ */
/*   /1* /2* PRINT(task.GetResults()); *2/ *1/ */
/*   /1* PRINT_VAR(arma::size(arma::find_nan(task.GetResults()))); *1/ */
/*   /1* auto lambdas = arma::linspace<arma::Row<DTYPE>>(0,1,10); *1/ */
/*   /1* task.Continue(lambdas); *1/ */
/*   /1* /2* PRINT(task.GetResults()); *2/ *1/ */
/*   /1* PRINT_VAR(arma::size(arma::find_nan(task.GetResults()))); *1/ */
/*   /1* task.Save("LCurve.bin"); *1/ */

/*   // Load Dataset */
/*   /1* Dataset data(212); *1/ */
/*   /1* PRINT_VAR(data.size_); *1/ */
/*   /1* data.Save("dataset.bin"); *1/ */
/*   /1* auto loaded = Dataset::Load(std::string("dataset.bin")); *1/ */
/*   /1* Dataset data2; *1/ */
/*   /1* data2 = std::move(*loaded); *1/ */
/*   /1* PRINT_VAR(arma::size(data2.inputs_)); *1/ */
/*   /1* PRINT_VAR(arma::size(data2.labels_)); *1/ */




/*   return 0; */
/* } */



