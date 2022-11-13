/**
 * @file leastsquares.h
 * @author Ozgur Taylan Turan
 *
 * Function that prepares data from input file arguments
 *
 *
 */


#ifndef LEAST_SQUARES_H 
#define LEAST_SQUARES_H 

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

#endif
