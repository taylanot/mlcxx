/**
 * @file leastsquaresregression.h
 * @author Ozgur Taylan Turan
 *
 * Function that prepares data from input file arguments for 
 * LeastSquaresRegression & KernelLeastSquaresRegression models
 *
 *
 */


#ifndef LEAST_SQUARES_REGRESSION_H 
#define LEAST_SQUARES_REGRESSION_H 

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
    auto inputs = trainset.inputs_;
    auto labels = arma::conv_to<arma::rowvec>::from(trainset.labels_);
    model_ = new mlpack::LinearRegression
      (inputs, labels, lambda_, bias_);
  };

  double Test(const utils::data::regression::Dataset& testset)
  {
    auto inputs = testset.inputs_;
    auto labels = arma::conv_to<arma::rowvec>::from(testset.labels_);
    return loss_.Evaluate(*model_, inputs, labels);
  };

 std::tuple<arma::mat, arma::mat>  
   LearningCurve(const utils::data::regression::Dataset& dataset,
                 jive::IdxVector N_bnds,
                 int N_res,
                 int repat)
  {
    auto inputs = dataset.inputs_;
    auto labels = arma::conv_to<arma::rowvec>::from(dataset.labels_);
    arma::irowvec Ns = 
        arma::regspace<arma::irowvec>(int(N_bnds[0]),int(N_bnds[1]), N_res_);
    if (tune_)
    {
        arma::vec lambdas = 
          arma::linspace(lambda_bnds_[0],lambda_bnds_[1],tune_res_);

        std::vector biases = {bias_bnds_[0], bias_bnds_[1]};
        src::regression::LCurve_HPT<mlpack::LinearRegression,
                        mlpack::MSE,
                        mlpack::SimpleCV> lcurve(Ns,repeat_,cv_valid_);
      
      lcurve.Generate(name_, inputs, labels, lambdas, biases);
      return lcurve.stats_;
    }

    else
    {

      src::regression::LCurve<mlpack::LinearRegression,
                        mlpack::MSE> lcurve(Ns,repeat_);
      
      lcurve.Generate(name_,inputs, labels, lambda_, bias_);
      return lcurve.stats_;
    }
  }

  void DoObjective(const jem::util::Properties objProps,
                   const utils::data::regression::Dataset& dataset)
  {
    BaseModel::DoObjective(objProps, dataset); 
  }

  private:
    mlpack::LinearRegression* model_;
    double lambda_;
    bool bias_;
    mlpack::MSE loss_;
    std::string name_ = "LeastSquaresRegression";

    bool              tune_;
    int               tune_res_;
    double            cv_valid_;
    jem::String       xval_type_;
    jive::Vector      lambda_bnds_;
    jive::BoolVector  bias_bnds_;
       
};

template<class T>
class KernelLeastSquaresRegression: public BaseModel
{
  public:

    KernelLeastSquaresRegression(const jem::util::Properties modelProps,
                     const utils::data::regression::Dataset& trainset) :
                     BaseModel(modelProps)
    {
      
      lambda_  = 0.;

      tune_res_ = 20;
      cv_valid_ = 0.2;
      xval_type_   = "SimpleCV";

      lambda_bnds_.resize(2); 
      lambda_bnds_[0] = 0.;
      lambda_bnds_[1] = 10.;

      tune_ = false;
      
      modelProps_.find( kernel_type_, "kernel.type" );

      kernel_param_bnds_.resize(2); 
      kernel_param_bnds_[0] = 1.e-6 ;
      kernel_param_bnds_[1] = 10.;

      
      modelProps_.find( tune_,   "hpt.tune" );
      modelProps_.find( lambda_, "hpt.lambda" );
         
      modelProps_.find( tune_res_,  "hpt.res" );
      modelProps_.find( xval_type_, "hpt.cv.type" );
      modelProps_.find( cv_valid_,  "hpt.cv.param" );
      
      modelProps_.find( kernel_param_bnds_, "kernel.param_bounds" );

      this -> Train(trainset);
    };

  void Train(const utils::data::regression::Dataset& trainset)
  {
    auto inputs = trainset.inputs_;
    auto labels = arma::conv_to<arma::rowvec>::from(trainset.labels_);
    model_ = new algo::regression::KernelRidge<T>
        (inputs, labels, lambda_);
  };

  double Test(const utils::data::regression::Dataset& testset)
  {
    auto inputs = testset.inputs_;
    auto labels = arma::conv_to<arma::rowvec>::from(testset.labels_);
    return loss_.Evaluate(*model_, inputs, labels);
  };

  std::tuple<arma::mat, arma::mat>  
   LearningCurve(const utils::data::regression::Dataset& dataset,
                 jive::IdxVector N_bnds,
                 int N_res,
                 int repat)
  {
    auto inputs = dataset.inputs_;
    auto labels = arma::conv_to<arma::rowvec>::from(dataset.labels_);

    arma::irowvec Ns = 
        arma::regspace<arma::irowvec>(int(N_bnds[0]),int(N_bnds[1]), N_res_);

    if (tune_)
    {
      arma::vec lambdas = 
                    arma::linspace(lambda_bnds_[0],lambda_bnds_[1],tune_res_);

      arma::vec params = arma::linspace(kernel_param_bnds_[0],
                                        kernel_param_bnds_[1], tune_res_);

      src::regression::LCurve_HPT<algo::regression::KernelRidge<T>,
                 mlpack::MSE,
                 mlpack::SimpleCV> lcurve(Ns,repeat_,cv_valid_);

      lcurve.Generate(name_, inputs, labels, lambdas, params);
      return lcurve.stats_;
    }
    else
    {

      src::regression::LCurve<algo::regression::KernelRidge<T>,
                        mlpack::MSE> lcurve(Ns,repeat_);
      
      lcurve.Generate(name_, inputs, labels, lambda_);
      return lcurve.stats_;
    }
  }

  void DoObjective(const jem::util::Properties objProps,
                   const utils::data::regression::Dataset& dataset)
  {
    BaseModel::DoObjective(objProps, dataset); 
  }

  private:
    algo::regression::KernelRidge<T>* model_;

    jem::String kernel_type_;

    double lambda_;
    
    mlpack::MSE loss_;
    std::string name_ = "KernelLeastSquaresRegression";

    bool              tune_;
    int               tune_res_;
    double            cv_valid_;
    jem::String       xval_type_;
    jive::Vector      lambda_bnds_;
    jive::Vector      kernel_param_bnds_;
       
};


#endif