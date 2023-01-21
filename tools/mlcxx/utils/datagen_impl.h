/**
 * @file datagen_impl.h
 * @author Ozgur Taylan Turan
 *
 * A simple toy data generation interface
 *
 * TODO: Increase the dimensionality, add some classification datasets banana??
 * TODO: Maybe add a base class for datagen
 *
 *
 */
#ifndef DATAGEN_IMPL_H
#define DATAGEN_IMPL_H

namespace utils {
namespace data {
namespace regression {

  Dataset::Dataset( ) { }
  Dataset::Dataset(size_t D, size_t N, double noise_std)
  {
    this -> size = N;
    this -> dimension = D;
    this -> noise_std = noise_std;
    
  }

  void Dataset::Generate(double scale, double phase, std::string type)
  {
    if (type == "Linear")
    {
      this -> inputs = (arma::randn(dimension,size) + phase);
      this -> labels = (inputs*scale + arma::randn(dimension,size)*noise_std);
    }
    else if (type == "Sine")
    {
      this -> inputs = (arma::randn(dimension,size) + phase);
      this -> labels = (scale * arma::sin(inputs)
                                        +arma::randn(dimension,size)*noise_std);
    }
  }

  void Dataset::Generate(size_t M, std::string type)
  {
    BOOST_ASSERT_MSG( dimension == 1, "FunctionalData input dim. != 1!");
    this -> labels.resize(M,size);  

    if (type == "SineFunctional")
    {
      this -> inputs = arma::sort(arma::randn(dimension,size),"ascend",1);
      arma::rowvec phase(M,arma::fill::randn);
      for( size_t i=0; i<M; i++ )
         this->labels.row(i) = arma::sin(inputs+phase(i))+
                                        arma::randn(dimension,size)*noise_std;
    }
  }

  void Dataset::Save(std::string filename)
  {
    arma::mat data = arma::join_cols(inputs, labels);
    utils::Save(filename, data, true);
  }

} // namespace regression
} // namespace data
} // namespace utils

namespace utils {
namespace data {
namespace regression {

  SineGen::SineGen( ) { }
  SineGen::SineGen(size_t M)
  {
    this -> size = M;

    this -> a = arma::rowvec(size, arma::fill::randn);
    this -> p = arma::rowvec(size, arma::fill::randn);
  }

  arma::mat SineGen::Predict(arma::mat inputs, std::string type) const 
  {
    arma::mat labels(size, inputs.n_cols);
    if (type == "Phase")
    {
      for (size_t i=0; i<size; i++)
        labels.row(i) = arma::sin(inputs+p(i));
      return labels;
    }
    else if (type == "Amplitude")
    {
      for (size_t i=0; i<size; i++)
        labels.row(i) = a(i) * arma::sin(inputs);
      return labels;
    }
    else if (type == "PhaseAmplitude")
    {
      for (size_t i=0; i<size; i++)
        labels.row(i) = a(i)*arma::sin(inputs+p(i));
      return labels;
    }
  }

} // namespace regression
} // namespace data
} // namespace utils


#endif
