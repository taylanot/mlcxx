/**
 * @file ae.cpp
 * @author Ozgur Taylan Turan
 *
 * Trying to develop an autoencoder for data generation
 *
 */
#define DTYPE double
#include <headers.h>


// Convenience typedefs
typedef mlpack::FFN<mlpack::ReconstructionLoss,
            mlpack::HeInitialization> ReconModel;

template <
    typename InputType = arma::mat,
    typename OutputType = arma::mat
>
class ReparametrizationType : public mlpack::Layer<InputType>
{
 public:
  ReparametrizationType(const bool stochastic = true,
                        const bool includeKl = true,
                        const double beta = 0.):
    stochastic(stochastic),
    includeKl(includeKl),
    beta(beta)

  {  }

  ReparametrizationType* Clone() const
  {
    return new ReparametrizationType(*this);
  }

  virtual ~ReparametrizationType() { }

  ReparametrizationType(const ReparametrizationType& layer){ };

  ReparametrizationType(ReparametrizationType&& layer) {};

  ReparametrizationType& operator=(const ReparametrizationType& layer){};

  ReparametrizationType& operator=(ReparametrizationType&& layer)
  {
    if (this != &layer)
    {
      mlpack::Layer<InputType>::operator=(std::move(layer));
      stochastic = std::move(layer.stochastic);
      includeKl = std::move(layer.includeKl);
      beta = std::move(layer.beta);
    }

    return *this;
  }


  void Forward(const InputType& input, OutputType& output)
  {
    const size_t latentSize = this->outputDimensions[0];
    mean = input.submat(latentSize, 0, 2 * latentSize - 1, input.n_cols - 1);
    preStdDev = input.submat(0, 0, latentSize - 1, input.n_cols - 1);

    if (stochastic)
      gaussianSample.randn(latentSize, input.n_cols);
    else
      gaussianSample.ones(latentSize, input.n_cols) * 0.7;

    mlpack::SoftplusFunction::Fn(preStdDev, stdDev);
    output = mean + stdDev % gaussianSample;
  }

  void Backward(const InputType& input,
                const OutputType& gy,
                OutputType& g)
  {
    OutputType tmp;
    mlpack::SoftplusFunction::Deriv(preStdDev, stdDev, tmp);

    if (includeKl)
    {
      g = arma::join_cols(gy % std::move(gaussianSample) % tmp + (-1 / stdDev + stdDev)
          % tmp * beta, gy + mean * beta / mean.n_cols);
    }
    else
    {
      g = arma::join_cols(gy % std::move(gaussianSample) % tmp, gy);
    }
  }
  double Loss()
  {
    if (!includeKl)
      return 0;

    return -0.5 * beta * accu(2 * log(stdDev) - pow(stdDev, 2)
        - pow(mean, 2) + 1) / mean.n_cols;
  }

  bool Stochastic() const { return stochastic; }
  bool& Stochastic() { return stochastic; }

  bool IncludeKL() const { return includeKl; }
  bool& IncludeKL() { return includeKl; }

  double Beta() const { return beta; }
  double& Beta() { return beta; }

  void ComputeOutputDimensions()
  {
    const size_t inputElem = std::accumulate(this->inputDimensions.begin(),
        this->inputDimensions.end(), 0);
    if (inputElem % 2 != 0)
    {
      std::ostringstream oss;
      oss << "Reparametrization layer requires that the total number of input "
          << "elements is divisible by 2!  (Received input with " << inputElem
          << " total elements.)";
      throw std::invalid_argument(oss.str());
    }

    this->outputDimensions = std::vector<size_t>(
        this->inputDimensions.size(), 1);
    // This flattens the input, and removes half the elements.
    this->outputDimensions[0] = inputElem / 2;
  }


  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

  private:
  //! If false, sample will be constant.
  bool stochastic;

  //! If false, KL error will not be included in Backward function.
  bool includeKl;

  //! The beta hyperparameter for constrained variational frameworks.
  double beta;

  //! Locally-stored current gaussian sample.
  OutputType gaussianSample;

  //! Locally-stored current mean.
  OutputType mean;

  //! Locally-stored pre standard deviation.
  //! After softplus activation gives standard deviation.
  OutputType preStdDev;

  //! Locally-stored current standard deviation.
  OutputType stdDev;
}; // class ReparametrizationType

// Standard Reparametrization layer.
typedef ReparametrizationType<arma::mat, arma::mat> Reparametrization;

/* int main() */
/* { */
/*   arma::mat data = arma::randu(1,10000); */
/*     ReconModel model; */
/*  // Encoder part. * */
/*   model.Add<mlpack::LinearType<arma::mat>>(100); */
/*   model.Add<mlpack::ReLU>(); */
/*   model.Add<mlpack::LinearType<arma::mat>>(25); */
/*   model.Add<mlpack::ReLU>(); */
/*   model.Add<mlpack::LinearType<arma::mat>>(10); */
/*   model.Add<mlpack::ReLU>(); */

/*   model.Add<ReparametrizationType<> >(2); */

/*   model.Add<mlpack::LinearType<arma::mat>>(10); */
/*   model.Add<mlpack::ReLU>(); */
/*   model.Add<mlpack::LinearType<arma::mat>>(25); */
/*   model.Add<mlpack::ReLU>(); */
/*   model.Add<mlpack::LinearType<arma::mat>>(100); */
/*   model.Add<mlpack::ReLU>(); */
/*   model.Add<mlpack::LinearType<arma::mat>>(1); */

/*   // Step 3: Train the model. */
/*   ens::Adam optimizer; // Default settings for Adam optimizer. */
  
/*   model.Train(data, data, optimizer); */

/*   // Step 4: Evaluate the model. */
/*   arma::mat reconstructed; */
/*   model.Predict(data, reconstructed); */

/*   // Calculate and print the reconstruction error. */
/*   double reconstructionError = arma::accu(arma::pow(reconstructed - data, 2)) / data.n_cols; */
/*   std::cout << "Reconstruction Error: " << reconstructionError << std::endl; */

/*   arma::mat u(10,1000,arma::fill::randn); */
/*   arma::mat out; */
/*   model.Forward(u,out,6,13); */

/*   return 0; */
  
/* } */

int main()
{
  // Step 1: Load or generate data.
  // For demonstration, we'll create a synthetic dataset.
  arma::mat data = arma::linspace(-10,10,10000).t();

  // Step 2: Define the autoencoder model.
  mlpack::FFN<mlpack::MeanSquaredErrorType<arma::Mat<DTYPE>>,
              mlpack::HeInitialization,
              arma::Mat<DTYPE>> model;
  
  // Encoder part.
  model.Add<mlpack::LinearType<arma::mat>>(25);
  model.Add<mlpack::ReLU>();
  model.Add<mlpack::LinearType<arma::mat>>(25);
  model.Add<mlpack::ReLU>();
  model.Add<mlpack::LinearType<arma::mat>>(10);
  model.Add<mlpack::ReLU>();
  model.Add<mlpack::LinearType<arma::mat>>(1);

  /* model.Add<mlpack::LinearType<arma::mat>>(10); */
  /* model.Add<mlpack::ReLU>(); */
  /* model.Add<mlpack::LinearType<arma::mat>>(25); */
  /* model.Add<mlpack::ReLU>(); */
  /* model.Add<mlpack::LinearType<arma::mat>>(100); */
  /* model.Add<mlpack::ReLU>(); */
  /* model.Add<mlpack::LinearType<arma::mat>>(1); */


  // Step 3: Train the model.
  ens::Adam optimizer; // Default settings for Adam optimizer.
  
  model.Train(data, arma::sin(data), optimizer);

  // Step 4: Evaluate the model.
  arma::mat reconstructed;
  arma::mat new_data;

  new_data.randu(1, 10000); // 10 features, 100 samples.
  model.Predict(new_data, reconstructed);
  arma::mat save1 = arma::join_vert(data,arma::sin(data));
  arma::mat save2 = arma::join_vert(new_data,arma::sin(new_data));
  save1.save("check1.csv",arma::csv_ascii);
  save2.save("check2.csv",arma::csv_ascii);

  // Calculate and print the reconstruction error.
  double reconstructionError = arma::accu(arma::pow(reconstructed - arma::sin(new_data), 2)) / data.n_cols;
  std::cout << "Reconstruction Error: " << reconstructionError << std::endl;

  return 0;
}
