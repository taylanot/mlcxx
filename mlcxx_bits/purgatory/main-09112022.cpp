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
#include "src/learning_curve.h"
#include "data/data_prep.h"
//#include "model/model.h"
//#include "register.h"

// standard
#include <variant>
#include <cstdio>
#include <vector>
#include <filesystem>

//
//auto ModelSelector(jem::String name)
//{
//  if (name == "Linear")
//  {
//    mlpack::regression::LinearRegression model;
//    return model;
//  }
//}


//auto DataProperties(jem::util::Properties& props)
//{
//  bool generate = true;
//
//  if (generate)
//  {
//    jem::String problem = "linear";
//    props.find(problem, "datagen.type");
//
//    if (problem == "linear")
//    {
//      int D = 1; double N = 1000; double a = 1.; double noise = 0.1;
//      props.find(D, "data.D");
//      props.find(N, "data.N");
//      props.find(a, "data.slope");
//      props.find(noise, "data.noise");
//      utils::datagen::regression::linear::dataset dataset(D, N, a, noise);
//      return dataset;
//    }
//    else if (problem == "nonlinear")
//    {
//      int D = 1; double N = 1000; double a = 1.; double p = 1.; 
//                                                 double noise = 0.1;
//      props.find(D, "data.D");
//      props.find(N, "data.N");
//      props.find(a, "data.amplitude");
//      props.find(p, "data.phase");
//      props.find(noise, "data.noise");
//      utils::datagen::regression::nonlinear::dataset dataset(D, N, a, p, noise);
//      return dataset;
//    }
//  }
//
//

const char*  DATA_PROP        = "data";
const char*  MODEL_PROP       = "model";
const char*  OBJECTIVE_PROP   = "objective";
//
//
//template<class T>
//auto DataPrepare(const T& dataProps)
//{
//  jem::String type = "regression"; jem::String genfunc; 
//  jem::String filename;  
//
//  bool generate = dataProps.find(genfunc,   "genfunc");
//  bool read     = dataProps.find(filename,  "filename");
//
//  if ((type == "regression") && generate)
//  { 
//    int D = 1; int Ntrn = 10; int Ntst = 1000;
//
//    double a = 1.; double p = 0.; double eps = 0.1;
//
//    dataProps.find(D,     "D");
//    dataProps.find(Ntrn,  "Ntrn");
//    dataProps.find(Ntst,  "Ntst");
//    dataProps.find(eps,   "eps");
//    dataProps.find(a,     "scale");
//    dataProps.find(p,     "phase");
//    dataProps.find(type,  "type");
//
//    std::cout << "*** Generating Data..." << std::endl;
//    std::cout << "  - D     : " << D  << "\n"
//              << "  - Ntrn  : " << Ntrn << "\n" 
//              << "  - Ntst  : " << Ntst << "\n" 
//              << "  - Scale : " << a << "\n" 
//              << "  - Phase : " << p << "\n" 
//              << "  - Noise : " << eps << "\n" 
//              << "  - From  : " << utils::to_string(genfunc) << std::endl;
//
//    utils::data::regression::Dataset trainset(D, Ntrn, eps);
//    utils::data::regression::Dataset testset(D, Ntst, eps);
//
//    trainset.Generate(a, p, utils::to_string(genfunc));
//    testset.Generate(a, p, utils::to_string(genfunc));
//    
//    return std::make_tuple(trainset, testset);
//  }
//
//}



//using Models = std::variant
//<
//  mlpack::regression::LinearRegression,
//  models::regression::KernelRidge<mlpack::kernel::GaussianKernel>
//>;
//
//template<class Type, class TrainData> 
//Models Training(const TrainData& data,
//                     const Type props)
//{
//  jem::String type;
//
//  jem::util::Properties modelProps = props.findProps(MODEL_PROP);
//  modelProps.get(type, "type");
//
//  bool tune = modelProps.contains("tune");
////=============================================================================
//// mlpack::regression::LinearRegression
////=============================================================================
//  if (type == "LinearRegression")
//  {
//
//    if (!tune)
//    {
//      double lambda = 0.; bool bias = true; 
//
//      std::cout << "Training with set hyperparameters..." << std::endl;
//      std::cout << "  - Type    : " << utils::to_string(type) << std::endl;
//
//      modelProps.find(lambda,  "lambda");
//      modelProps.find(bias,    "bias");
//
//      std::cout << "  - Lambda  : " << lambda << "\n" 
//                << "  - Bias    : " << bias   << std::endl;
//
//      mlpack::regression::LinearRegression model(data.inputs, data.labels,
//                                                 lambda, bias);
//      return model;
//    }
//
//    else
//    {
//
//      double best_lambda; bool best_bias;
//
//      std::cout << "*** Searching hyperparameters..." << std::endl;
//
//      jem::util::Properties tuneProps = modelProps.findProps("tune"); 
//      jive::Vector lambda; tuneProps.find(lambda, "hyper.lambda");
//      jive::BoolVector bias; tuneProps.find(bias, "hyper.bias");
//
//      int res = 100; tuneProps.find(res, "res");
//
//      arma::vec lambdas = arma::linspace(lambda[0],lambda[1],res);
//      std::vector biases = {bias[0], bias[1]};
//      
//
//      jem::String cv_type = "SimpleCV"; tuneProps.find(cv_type, "cv.type");
//      jem::String loss = "MSE"; modelProps.find(loss, "loss");
//
//      std::cout << "  - CV    : " << utils::to_string(cv_type) << "\n"
//                << "  - LOSS  : " << utils::to_string(loss) << std::endl;
//
//
//      if (cv_type == "SimpleCV" && loss == "MSE")
//      {
//      double valid = 0.2; tuneProps.find(valid, "cv.valid");
//
//      mlpack::hpt::HyperParameterTuner
//      <mlpack::regression::LinearRegression,
//       mlpack::cv::MSE,
//       mlpack::cv::SimpleCV> hpt(valid, data.inputs, data.labels);
//      std::tie(best_lambda, best_bias) = hpt.Optimize(lambdas,biases);
//
//      }
//      else if (cv_type == "KFoldCV" && loss == "MSE")
//      {
//
//      int fold= 5; tuneProps.find(fold, "cv.fold");
//
//      mlpack::hpt::HyperParameterTuner
//      <mlpack::regression::LinearRegression,
//       mlpack::cv::MSE,
//       mlpack::cv::KFoldCV> hpt(fold, data.inputs, data.labels);
//      std::tie(best_lambda, best_bias) = hpt.Optimize(lambdas,biases);
//      }
//      
//      std::cout << "*** Training with best hyperparameters..." << std::endl;
//      std::cout << "  - Lambda  : " << best_lambda << "\n" 
//                << "  - Bias    : " << best_bias   << std::endl;
//
//      mlpack::regression::LinearRegression model(data.inputs, data.labels,
//                                                 best_lambda, best_bias);
//      return model;
//    }
//
//  }
////=============================================================================
//// models::regression::KernelRidge<mlpack::kernel::GaussianKernel>
////=============================================================================
//  else if (type == "GaussianKernelRidge")
//  {
//    if (!tune)
//    {
//    double lambda = 0.; double l = 1.;
//
//    modelProps.find(lambda,  "lambda");
//    modelProps.find(l,       "l");
//
//    std::cout << "  - Lambda      : " << lambda << "\n" 
//              << "  - LengthScale : " << l      << std::endl;
//
//    models::regression::KernelRidge<mlpack::kernel::GaussianKernel> 
//      model(data.inputs, data.labels, lambda, l);
//    return model;
//    }
//    else
//    {
//
//      double best_lambda; bool best_l;
//
//      std::cout << "*** Searching hyperparameters..." << std::endl;
//
//      jem::util::Properties tuneProps = modelProps.findProps("tune"); 
//      jive::Vector lambda; tuneProps.find(lambda, "hyper.lambda");
//      jive::Vector l; tuneProps.find(l, "hyper.l");
//
//      int res = 100; tuneProps.find(res, "res");
//
//      arma::vec lambdas = arma::linspace(lambda[0],lambda[1],res);
//      arma::vec ls = arma::linspace(l[0],l[1],res);
//      
//
//      jem::String cv_type = "SimpleCV"; tuneProps.find(cv_type, "cv.type");
//      jem::String loss = "MSE"; modelProps.find(loss, "loss");
//
//      std::cout << "  - CV    : " << utils::to_string(cv_type) << "\n"
//                << "  - LOSS  : " << utils::to_string(loss) << std::endl;
//
//
//      if (cv_type == "SimpleCV" && loss == "MSE")
//      {
//        double valid = 0.2; tuneProps.find(valid, "cv.valid");
//
//        mlpack::hpt::HyperParameterTuner
//        <models::regression::KernelRidge<mlpack::kernel::GaussianKernel>,
//         mlpack::cv::MSE,
//         mlpack::cv::SimpleCV> hpt(valid, data.inputs, data.labels);
//        std::tie(best_lambda, best_l) = hpt.Optimize(lambdas,ls);
//
//      }
//      else if (cv_type == "KFoldCV" && loss == "MSE")
//      {
//
//        int fold= 5; tuneProps.find(fold, "cv.fold");
//
//        mlpack::hpt::HyperParameterTuner
//        <models::regression::KernelRidge<mlpack::kernel::GaussianKernel>,
//         mlpack::cv::MSE,
//         mlpack::cv::KFoldCV> hpt(fold, data.inputs, data.labels);
//        std::tie(best_lambda, best_l) = hpt.Optimize(lambdas,ls);
//      }
//      
//      std::cout << "*** Training with best hyperparameters..." << std::endl;
//      std::cout << "  - Lambda      : " << best_lambda << "\n" 
//                << "  - Lengthscale : " << best_l << std::endl;
//
//      models::regression::KernelRidge<mlpack::kernel::GaussianKernel> 
//                      model(data.inputs, data.labels, best_l, best_lambda);
//      return model;
//    }
//  }
//  return {};
//}
//
//template<class T>
//double MSETestError(T& datasets, jem::util::Properties props)
//{
//  auto Predict = [&datasets] (const auto& obj) 
//  {
//    arma::rowvec pred;
//    obj.Predict(std::get<1>(datasets).inputs, pred);
//    auto labels = arma::conv_to<arma::rowvec>::from
//                                                (std::get<1>(datasets).labels);
//    auto inputs = std::get<1>(datasets).inputs; 
//    double error = mlpack::cv::MSE::Evaluate(obj,inputs, labels);
//    std::cout << "  - Test Loss : " 
//              << error << std::endl;
//    return error;
//  };
//  return std::visit(Predict, Training(std::get<0>(datasets), props));
//}
//
//void run(jem::util::Properties& props)
//{
//
//  jem::String objective;
//  props.get(objective,"objective.type");
//
//  jem::util::Timer timer;
//  timer.start(); 
//  auto datasets = DataPrepare(props);
//  std::cout << " [ Data Preparation Elapsed Time : "  << timer.toDouble() 
//                                                       << " ]" << std::endl;
//  std::cout.put('\n');
//
//  timer.start(); 
//  if (objective == "train/test")
//  {
//    double error = MSETestError(datasets, props);
//        std::cout.put('\n');
//  }
//  std::cout << " [ Objective : " << utils::to_string(objective) <<
//               " Elapsed Time : " << timer.toDouble() << " ]"  << std::endl;
//
//  //else
//  //{
//  //  std::cerr << "Nothing to do: Define a problem! " << std::endl;
//  //}
//  //std::cout << err << std::endl
//  //models::regression::KernelRidge<mlpack::kernel::GaussianKernel> model(std::get<0>(datasets).inputs, std::get<0>(datasets).labels,0.4,1.);
//  //mlpack::regression::LinearRegression  model(std::get<0>(datasets).inputs, std::get<0>(datasets).labels);
//  //auto inputs = std::get<0>(datasets).inputs;
//  //arma::rowvec pred_labels; 
//  //model.Predict(inputs,pred_labels);
//  //std::cout << pred_labels << std::endl;  
//  //auto caller = [dataset.inputs&, dataset.labels&](const auto& obj) { obj.Train(dataset.inputs,dataset.labels); }
//  //std::visit(TrainModels{}, Train(type, dataset.inputs, dataset.labels));
//  //models::regression::KernelRidge<mlpack::kernel::GaussianKernel> model(dataset.inputs,dataset.labels,0.4,1.);
//
//
//  //arma::mat pred_inputs = arma::linspace<arma::rowvec>(-2,2,10);
//  //arma::rowvec pred_labels; 
//  //auto caller = [&pred_inputs, &pred_labels] (const auto& obj) { obj.Predict(pred_inputs, pred_labels); };
//  //std::visit(caller, Train(type, dataset.inputs, dataset.labels));
//  //std::cout << pred_labels << std::endl;
//  
//
//  //std::visit(Predict, model);
//  //std::cout << pred_labels << std::endl;
//  //utils::covmat<mlpack::kernel::GaussianKernel> cov(1);
//  //std::cout << cov.GetMatrix(dataset.inputs.t(), dataset.inputs.t()) << std::endl; 
//
//  
//  //std::cout << dataset.inputs << std::endl;
//  //FOO<int> ali(1);
//  //models::regression::KernelRidge<mlpack::kernel::GaussianKernel> model(dataset.inputs,dataset.labels,0.4,1.);
//  //std::unique_ptr<Model> model;  
//  //if (type == 1)  
//  //{
//  //    model.reset(new models::regression::KernelRidge<mlpack::kernel::GaussianKernel>(dataset.inputs,dataset.labels,0.4,1.));
//  //    std::cout << model->Parameters() << std::endl;
//  //}
//  //else
//  //    model.reset(new models::regression::KernelRidge<mlpack::kernel::GaussianKernel>(dataset.inputs,dataset.labels,0.4,1.));
//
//  //models::regression::KernelRidge<mlpack::kernel::GaussianKernel> ml(dataset.inputs,dataset.labels,0.4,1.);
//  
//  //pred_inputs.save("pred_inputs.csv",arma::file_type::csv_ascii);
//  //pred_labels.save("pred_labels.csv",arma::file_type::csv_ascii);
//  //inputs.save("inputs.csv",arma::file_type::csv_ascii);
//  //labels.save("labels.csv",arma::file_type::csv_ascii);
//  
//  //std::unique_ptr<Hardware> hw;  
//  //if (type == 1)  
//  //    hw.reset(new Hardware1(1));
//  //else
//  //    hw.reset(new Hardware2(1,2));
//
//  //hw->doSomething();
//
//  //utils::data::regression::Dataset dataset(D, N, noise);
//  //dataset.Generate(a, p, utils::to_string(type));
//  //std::cout << dataset.inputs << std::endl;
//  //std::cout << dataset.labels<< std::endl;
//
//
//  //auto dataset = dataProperties(props);
//  //std::cout << dataset.inputs << std::endl;
//  //auto model = ModelSelector(model_name); 
//  //using models = std::variant<mlpack::regression::LinearRegression, mlpack::regression::KernelRidge>;
//
//  //struct ModelTable{
//  //models modelkkk
//  //};
//
//
//
//  //ModelRegister<1>::modelClass model(trainset.inputs,trainset.labels,0.0,1.0);
//  //mymodels["linear"] model(trainset.inputs.t(),trainset.labels);
//  ////jem::System::out() << N << jem::io::endl;
//  ////jem::System::out() << ampl << jem::io::endl;
//  //std::cout << N << std::endl;
//  //std::cout << int(props.size()) << std::endl;
//  //std::cout << ampl << std::endl;
//
//}
//
//void run_test()
//{
//    int D = 1; int Ntrn = 100; int Ntst = 1000;
//
//    double a = 1.; double p = 0.0; 
//
//    double noise = .1;
//
//    jem::String type = "Sine";
//    utils::data::regression::Dataset trainset(D, Ntrn, noise);
//    utils::data::regression::Dataset testset(D, Ntst, noise);
//    trainset.Generate(a, p, utils::to_string(type));
//    testset.Generate(a, p, utils::to_string(type));
//
//    arma::mat inputs = trainset.inputs;
//    arma::mat labels = trainset.labels;
//
//    double valid = 0.2;
//    mlpack::hpt::HyperParameterTuner
//      <models::regression::KernelRidge<mlpack::kernel::GaussianKernel>,
//       mlpack::cv::MSE, mlpack::cv::SimpleCV> hpt(valid, inputs, labels);
//    arma::vec lambdas=arma::linspace(0,1,100);
//    arma::vec ls=arma::linspace(0.000001,1,100);
//    double best_lambda, best_l;
//    std::tie(best_lambda, best_l) = hpt.Optimize(lambdas,ls);
//
//    std::cout << best_lambda << std::endl;
//    std::cout << best_l << std::endl;
//
//    trainset.Save("trainset.csv");
//    testset.Save("testset.csv");
//
//    auto  best_ = hpt.BestModel();
//
//    arma::mat pred_inps = arma::sort(testset.inputs,"ascent",1);
//    arma::rowvec pred_labels;
//    best_.Predict(pred_inps, pred_labels);
//    std::string filename = "prediction.csv";
//    arma::mat pred = arma::conv_to<arma::mat>::from(pred_labels);
//    utils::Save(filename,pred_inps,pred);
//
//}

//template<class MODEL,
//        class LOSS>
//class BaseModel 
//{
//  public:
//    template<class Input, class Label, class... ARGS>
//    BaseModel<MODEL, LOSS>(Input inputs, Label labels, ARGS... args) :
//                                            model_(inputs, labels, args...) {};
//
//    BaseModel<MODEL, LOSS>() {};
//
//    MODEL model_;
//    LOSS loss_;
//
//  public:
//    double Test(const utils::data::regression::Dataset& testset)
//    {
//      auto inputs = testset.inputs;
//      auto labels = arma::conv_to<arma::rowvec>::from(testset.labels);
//      return loss_.Evaluate(model_, inputs, labels);
//    }
//
//    //virtual void LearningCurve() = 0;
//    //virtual void FeatureCurve() = 0;
//};
//
//class LeastSquaresRegression: public 
//                           BaseModel<mlpack::regression::LinearRegression,
//                                     mlpack::cv::MSE>
//{
//  public:
//    LeastSquaresRegression ( ) { };
//    template<class Input, class Label, class... ARGS>
//    LeastSquaresRegression (const Input& inputs, const Label& labels,
//                            ARGS... args) :
//             BaseModel<mlpack::regression::LinearRegression,
//                       mlpack::cv::MSE> (inputs, labels, args...) {};
//
//  double Test(const utils::data::regression::Dataset& testset)
//  {
//    return BaseModel::Test(testset);
//  };
//
//  double LearningCurve(const utils::data::regression::Dataset& testset)
//  {
//    return BaseModel::Test(testset);
//  }
//};

class BaseModel 
{

  public:

    BaseModel() {};

    BaseModel(const jem::util::Properties modelProps) : modelProps_(modelProps)
                                                                           {};
    
    virtual void Train(const utils::data::regression::Dataset& trainset) = 0;

    virtual double Test(const utils::data::regression::Dataset& testset) = 0;

    virtual std::tuple<arma::mat, arma::mat> LearningCurve
                        (const utils::data::regression::Dataset& dataset, 
                         const jive::IdxVector N_bnds,
                         const int N_res,
                         const int repat) = 0;

    void DoObjective(const jem::util::Properties objProps,
                     const utils::data::regression::Dataset& dataset)
    {
      jem::String obj;
      
      objProps.find(obj,"type");

      if ( obj == "learningcurve" )
      {
        N_res_  = 20;
        repeat_ = 100;

        N_bnds_.resize(2); 
        N_bnds_[0] = 5; 
        N_bnds_[1] = 100;
        
        objProps.find( repeat_, "rep"      );
        objProps.find( N_bnds_, "N_bounds" );
        objProps.find( N_res_,  "res"      );

        this -> LearningCurve(dataset, N_bnds_, N_res_, repeat_);
      }
      else if ( obj == "train/test" )
      {
        double err = this -> Test(dataset);
        std::cout << err << std::endl;
      }
    };

  protected:

    int                     N_res_;
    int                     repeat_;
    jive::IdxVector         N_bnds_;
    jem::util::Properties   modelProps_;

};


class LeastSquaresRegression: public BaseModel
{
  public:

    LeastSquaresRegression(const jem::util::Properties modelProps,
                           const utils::data::regression::Dataset& trainset) :
                           BaseModel(modelProps)
    {
      
      lambda_  = 0.;
      bias_    = true;

      tune_res_ = 20;
      cv_valid_ = 0.2;
      xval_type_   = "SimpleCV";

      lambda_bnds_.resize(2); 
      lambda_bnds_[0] = 0.;
      lambda_bnds_[1] = 10.;

      tune_ = false;

      bias_bnds_.resize(2); 
      bias_bnds_[0] = true ;
      bias_bnds_[1] = false;

      modelProps_.find( tune_,   "hpt.tune" );
      modelProps_.find( bias_,   "hpt.bias" );
      modelProps_.find( lambda_, "hpt.lambda" );
         
      modelProps_.find( tune_res_,  "hpt.res" );
      modelProps_.find( xval_type_, "hpt.cv.type" );
      modelProps_.find( cv_valid_,  "hpt.cv.param" );
      
      modelProps_.find( bias_bnds_,   "hpt.bias_bounds" );
      modelProps_.find( lambda_bnds_, "hpt.lambda_bounds" );


      this -> Train(trainset);
    };

  void Train(const utils::data::regression::Dataset& trainset)
  {
      model_ = new mlpack::regression::LinearRegression
        (trainset.inputs, trainset.labels, lambda_, bias_);
  };

  double Test(const utils::data::regression::Dataset& testset)
  {
    auto inputs = testset.inputs;
    auto labels = arma::conv_to<arma::rowvec>::from(testset.labels);
    return loss_.Evaluate(*model_, inputs, labels);
  };

 std::tuple<arma::mat, arma::mat>  
   LearningCurve(const utils::data::regression::Dataset& dataset,
                 jive::IdxVector N_bnds,
                 int N_res,
                 int repat)
  {
    arma::irowvec Ns = 
        arma::regspace<arma::irowvec>(int(N_bnds[0]),int(N_bnds[1]), N_res_);
    if (tune_)
    {
        arma::vec lambdas = 
          arma::linspace(lambda_bnds_[0],lambda_bnds_[1],tune_res_);

        std::vector biases = {bias_bnds_[0], bias_bnds_[1]};

        LCurve_HPT<mlpack::regression::LinearRegression,
                        mlpack::cv::MSE,
                        mlpack::cv::SimpleCV> lcurve(name_,Ns,repeat_,cv_valid_);
      
      return  lcurve.Generate(dataset.inputs, dataset.labels, lambdas, biases);
    }

    else
    {

      LCurve<mlpack::regression::LinearRegression,
                        mlpack::cv::MSE> lcurve(name_,Ns,repeat_);
      
      return  lcurve.Generate(dataset.inputs, dataset.labels, lambda_, bias_);
    }
  }

  void DoObjective(const jem::util::Properties objProps,
                   const utils::data::regression::Dataset& dataset)
  {
    BaseModel::DoObjective(objProps, dataset); 
  }

  private:
    mlpack::regression::LinearRegression* model_;
    double lambda_;
    bool bias_;
    mlpack::cv::MSE loss_;
    std::string name_ = "LeastSquaresRegression";

    bool              tune_;
    int               tune_res_;
    double            cv_valid_;
    jem::String       xval_type_;
    jive::Vector      lambda_bnds_;
    jive::BoolVector  bias_bnds_;
       
};


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
  if (true)
  {
    model_ptr = new LeastSquaresRegression(modelProps, std::get<0>(datasets));
  }

  model_ptr -> DoObjective(objProps, std::get<0>(datasets));
  std::cout << " [ Objective Elapsed Time : "  << timer.toDouble() 
                                                       << " ]" << std::endl;


}


void test()
{
  std::cout << "test" << std::endl;
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


//int run_oldtests()
//{
//  size_t seed = 24;  // KOBEE!!
//  arma::arma_rng::set_seed(seed);
//  mlpack::math::RandomSeed(seed);
//
//  //jem::String mode = "test_saving_plotting";
//  //jem::String mode = "test_learning_curve_lin_nohpt";
//  jem::String mode = "test_learning_curve_nonlin_nohpt";
//  //jem::String mode = "test_kernelridge";
//
//  if (mode == "test_kernelridge")
//  {
//    int D, Ntrn, Ntst; D=1; Ntrn=50; Ntst=1000;
//    double ampl, phase, noise_std; ampl= 1.0; phase=0.; noise_std = 0.1;
//    utils::datagen::regression::nonlinear::dataset trainset(Ntrn,
//                                                             ampl,
//                                                             phase,
//                                                             noise_std);
//    arma::mat inputs = trainset.inputs;
//    arma::mat labels = trainset.labels;
//
//    //arma::mat inputs = arma::linspace<arma::rowvec>(-5,5,Ntrn);
//    //arma::mat labels = arma::sin(inputs);
//    //mlpack::regression::KernelRidge<mlpack::kernel::GaussianKernel> model(inputs.t(),labels,0.4,1.);
//    //arma::mat pred_inputs = arma::linspace<arma::rowvec>(-2,2,Ntst);
//    //arma::rowvec pred_labels;
//    //model.Predict(pred_inputs.t(), pred_labels,1.);
//    //pred_inputs.save("pred_inputs.csv",arma::file_type::csv_ascii);
//    //pred_labels.save("pred_labels.csv",arma::file_type::csv_ascii);
//    //inputs.save("inputs.csv",arma::file_type::csv_ascii);
//    //labels.save("labels.csv",arma::file_type::csv_ascii);
//  }
//  else if (mode == "test_hypertune_linear")
//  {
//    int D, Ntrn, Ntst; D=1; Ntrn=4; Ntst=1000;
//    double ampl, phase, noise_std; ampl= 1.0; phase=0.; noise_std = 0.1;
//    utils::datagen::regression::linear::dataset trainset(Ntrn,
//                                                             ampl,
//                                                             noise_std);
//    arma::mat inputs = trainset.inputs;
//    arma::mat labels = trainset.labels;
//
//    double valid = 0.2;
//    mlpack::hpt::HyperParameterTuner<mlpack::regression::LinearRegression, mlpack::cv::MSE, mlpack::cv::SimpleCV> hpt(valid, inputs, labels);
//    arma::vec lambdas{0.,0.01, 0.001};
//    double best;
//    std::tie(best) = hpt.Optimize(lambdas);
//
//    std::cout << best << std::endl;
//      
//    //arma::mat pred_inputs = arma::linspace<arma::rowvec>(-2,2,Ntst);
//    //arma::rowvec pred_labels;
//    //model.Predict(pred_inputs.t(), pred_labels,1.);
//    //pred_inputs.save("pred_inputs.csv",arma::file_type::csv_ascii);
//    //pred_labels.save("pred_labels.csv",arma::file_type::csv_ascii);
//    //inputs.save("inputs.csv",arma::file_type::csv_ascii);
//    //labels.save("labels.csv",arma::file_type::csv_ascii);
//
//  }
//  else if (mode == "test_hypertune_kernel")
//  {
//    int D, Ntrn, Ntst; D=1; Ntrn=100; Ntst=1000;
//    double ampl, phase, noise_std; ampl= 1.0; phase=0.; noise_std = 0.1;
//    utils::datagen::regression::nonlinear::dataset trainset(Ntrn,
//                                                             ampl,
//                                                             phase, 
//                                                             noise_std);
//    arma::mat inputs = trainset.inputs;
//    arma::mat labels = trainset.labels;
//
//    double valid = 2.;
//    mlpack::hpt::HyperParameterTuner
//      <mlpack::regression::KernelRidge<mlpack::kernel::GaussianKernel>,
//       mlpack::cv::MSE, mlpack::cv::SimpleCV> hpt(valid, inputs, labels);
//    arma::vec lambdas=arma::linspace(0,1,100);
//    arma::vec ls=arma::linspace(0.000001,1,100);
//    double best_lambda, best_l;
//    std::tie(best_lambda, best_l) = hpt.Optimize(lambdas,ls);
//
//    std::cout << best_lambda << std::endl;
//    std::cout << best_l << std::endl;
//      
//    //arma::mat pred_inputs = arma::linspace<arma::rowvec>(-2,2,Ntst);
//    //arma::rowvec pred_labels;
//    //model.Predict(pred_inputs.t(), pred_labels,1.);
//    //pred_inputs.save("pred_inputs.csv",arma::file_type::csv_ascii);
//    //pred_labels.save("pred_labels.csv",arma::file_type::csv_ascii);
//    //inputs.save("inputs.csv",arma::file_type::csv_ascii);
//    //labels.save("labels.csv",arma::file_type::csv_ascii);
//
//  }
//  else if (mode == "test_learning_curve_nonlin")
//  {
//    int D, Ntrn, Ntst; D=1; Ntrn=100; Ntst=1000;
//    double ampl, phase, noise_std; ampl= 1.0; phase=0.; noise_std = 0.;
//    utils::datagen::regression::nonlinear::dataset trainset(Ntrn,
//                                                             ampl,
//                                                             phase, 
//                                                             noise_std);
//    arma::mat inputs = trainset.inputs;
//    arma::rowvec labels = trainset.labels;
//    double tune_res = 20;
//    double split_ratio = 0.2;
//    double tune_arg = 0.2;
//    size_t repeat = 100;
//    
//    arma::vec lambdas = arma::linspace(0,10,tune_res);
//    arma::vec ls = arma::linspace(0.01,10,tune_res);
//    arma::irowvec Ns = arma::regspace<arma::irowvec>(5, 1, 20);
//
//    LearningCurve_HPT<mlpack::regression::KernelRidge
//      <mlpack::kernel::GaussianKernel>, mlpack::cv::MSE, mlpack::cv::SimpleCV> 
//                                                    lcurve(Ns,repeat,tune_arg);
//    lcurve.Generate(inputs, labels, lambdas, ls);
//    std::tuple<arma::mat, arma::mat> res = lcurve.Generate(inputs, labels, lambdas, ls);//, tune_arg, Ns, model);
//    arma::mat trn = std::get<0>(res);
//    arma::mat tst = std::get<1>(res);
//    arma::mat results = arma::join_cols(arma::conv_to<arma::rowvec>::from(Ns), trn, tst);
//    std::string filename = "learning_curve.csv";
//    utils::Save(filename,results);
//  }
//  else if (mode == "test_learning_curve_nonlin_nohpt")
//  {
//    int D, Ntrn, Ntst; D=1; Ntrn=500; Ntst=1000;
//    double ampl, phase, noise_std; ampl= 1.0; phase=0.; noise_std = 1.;
//    utils::datagen::regression::nonlinear::dataset trainset(Ntrn,
//                                                             ampl,
//                                                             phase, 
//                                                             noise_std);
//    arma::mat inputs = trainset.inputs;
//    arma::rowvec labels = trainset.labels;
//    size_t repeat = 500;
//    
//    double lambda = 0.5;
//    double l = 1.;
//    arma::irowvec Ns = arma::regspace<arma::irowvec>(10, 20, 400);
//
//    LearningCurve<mlpack::regression::KernelRidge
//      <mlpack::kernel::GaussianKernel>, mlpack::cv::MSE> lcurve(Ns,repeat);
//
//    std::tuple<arma::mat, arma::mat> res = lcurve.Generate(inputs, labels, lambda, l);//, tune_arg, Ns, model);
//    arma::mat trn = std::get<0>(res);
//    arma::mat tst = std::get<1>(res);
//    arma::mat results = arma::join_cols(arma::conv_to<arma::rowvec>::from(Ns), trn, tst);
//    std::string filename = "learning_curve.csv";
//    utils::Save(filename,results);
//  }
//  else if (mode == "test_learning_curve_lin")
//  {
//    int D, Ntrn, Ntst; D=1; Ntrn=100; Ntst=1000;
//    double ampl, phase, noise_std; ampl= 1.0; phase=0.; noise_std = 0.;
//    utils::datagen::regression::linear::dataset trainset(Ntrn,
//                                                             ampl,
//                                                             noise_std);
//    arma::mat inputs = trainset.inputs;
//    arma::rowvec labels = trainset.labels;
//    double tune_res = 20;
//    double tune_arg = 0.2;
//    size_t repeat = 100;
//    
//    arma::vec lambdas = arma::linspace(0,10,tune_res);
//    arma::irowvec Ns = arma::regspace<arma::irowvec>(5, 1, 100);
//
//    LearningCurve_HPT<mlpack::regression::LinearRegression,
//                  mlpack::cv::MSE, mlpack::cv::SimpleCV> 
//                                                    lcurve(Ns,repeat,tune_arg);
//    //lcurve.Generate(inputs, labels, lambdas);
//    std::tuple<arma::mat, arma::mat> res = lcurve.Generate(inputs, labels, lambdas);//, tune_arg, Ns, model);
//    arma::mat trn = std::get<0>(res);
//    arma::mat tst = std::get<1>(res);
//    arma::mat results = arma::join_cols(arma::conv_to<arma::rowvec>::from(Ns), trn, tst);
//    std::string filename = "learning_curve.csv";
//    utils::Save(filename,results);
//  }
//  else if (mode == "test_learning_curve_lin_nohpt")
//  {
//    int D, Ntrn, Ntst; D=1; Ntrn=100; Ntst=1000;
//    double ampl, phase, noise_std; ampl= 1.0; phase=0.; noise_std = 1.;
//    utils::datagen::regression::linear::dataset trainset(Ntrn,
//                                                         ampl,
//                                                         noise_std);
//    arma::mat inputs = trainset.inputs;
//    arma::rowvec labels = trainset.labels;
//
//    size_t repeat = 1000;
//
//    
//    arma::irowvec Ns = arma::regspace<arma::irowvec>(5, 10, 100);
//
//    LearningCurve<mlpack::regression::LinearRegression,
//                  mlpack::cv::MSE> lcurve(Ns,repeat);
//
//    double lambda = 0.;
//    //lcurve.Generate(inputs, labels, lambda);
//    std::tuple<arma::mat, arma::mat> res = lcurve.Generate(inputs, labels, lambda);//, tune_arg, Ns, model);
//    arma::mat trn = std::get<0>(res);
//    arma::mat tst = std::get<1>(res);
//    arma::mat results = arma::join_cols(arma::conv_to<arma::rowvec>::from(Ns), trn, tst);
//    std::string filename = "learning_curve.csv";
//    utils::Save(filename,results);
//  }
//  else if (mode == "test_saving_plotting")
//  {
//    int D, Ntrn, Ntst; D=1; Ntrn=200; Ntst=1000;
//    double ampl, phase, noise_std; ampl= 1.0; phase=0.; noise_std = 0.0;
//    utils::datagen::regression::nonlinear::dataset trainset(Ntrn,
//                                                            ampl,
//                                                            phase, 
//                                                            noise_std);
//
//    utils::datagen::regression::nonlinear::dataset testset(Ntst,
//                                                            ampl,
//                                                            phase, 
//                                                            noise_std);
//    arma::mat inputs = trainset.inputs;
//    arma::rowvec labels = trainset.labels;
//    trainset.save("trainset.csv");
//    testset.save("testset.csv");
//
//    double valid = 0.2;
//    mlpack::hpt::HyperParameterTuner
//      <mlpack::regression::KernelRidge<mlpack::kernel::GaussianKernel>,
//       mlpack::cv::MSE, mlpack::cv::SimpleCV> hpt(valid, inputs, labels);
//    arma::vec lambdas=arma::linspace(0,10,100);
//    arma::vec ls=arma::linspace(0.001,1,100);
//    double best_lambda, best_l;
//    std::tie(best_lambda, best_l) = hpt.Optimize(lambdas,ls);
//    std::cout << best_lambda << std::endl;
//    std::cout << best_l << std::endl;
//    mlpack::regression::KernelRidge<mlpack::kernel::GaussianKernel> best_ = 
//                                                                hpt.BestModel();
//
//    arma::mat pred_inps = arma::sort(testset.inputs,"ascent",1);
//    arma::rowvec pred_labels;
//    best_.Predict(pred_inps, pred_labels,best_l);
//    std::string filename = "prediction.csv";
//    arma::mat pred = arma::conv_to<arma::mat>::from(pred_labels);
//    utils::Save(filename,pred_inps,pred);
//  }
//  return 0;
//}
//
//int main ()
//{
//  jem::util::Timer timer;
//  timer.start();
//  run();
//  std::cout << "CPU Time Spent : " << timer.toDouble() << std::endl;
//}

  
