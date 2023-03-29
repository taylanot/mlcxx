/**
 * @file datagen_impl.h
 * @author Ozgur Taylan Turan
 *
 * A simple toy data generation interface
 *
 * TODO: Increase the dimensionality, add some classification datasets banana??
 * TODO: Maybe add a base class for datagen
 * TODO: Maybe Functional data generation can be done with each_row()
 *
 *
 */
#ifndef DATAGEN_IMPL_H
#define DATAGEN_IMPL_H

namespace utils {
namespace data {
namespace regression {

  Dataset::Dataset ( ) { }
  Dataset::Dataset ( const size_t& D, const size_t& N ) :
                     size_(N), dimension_(D) { }

  void Dataset::Set ( const size_t& D, const size_t& N)
  {
    size_ = N;
    dimension_ = D;
  }

  void Dataset::Generate ( const double& scale, const double& phase,
                           const std::string& type )
  {
    if (type == "Linear")
    {
      inputs_ = arma::randn(dimension_,size_) + phase;
      labels_ = (scale*arma::ones(dimension_)).t() * inputs_;
    }
    else if (type == "Sine")
    {
      inputs_ = arma::randn(dimension_,size_) + phase;
      labels_ = arma::sin((scale*arma::ones(dimension_)).t() * inputs_);
    }
  }

  void Dataset::Generate ( const double& scale, const double& phase,
                           const std::string& type, const double& noise_std )
  {
      this -> Generate(scale, phase, type);
      this -> Noise(noise_std);
  }

  void Dataset::Generate ( const size_t& M, const std::string& type )
  {
    BOOST_ASSERT_MSG( dimension_ == 1, "FunctionalData input dim. != 1!");
    labels_.resize(M,size_);  

    if (type == "SineFunctional")
    {
      inputs_ = arma::sort(arma::randn(dimension_,size_),"ascend",1);
      arma::rowvec phase(M,arma::fill::randn);
      for( size_t i=0; i<M; i++ )
         labels_.row(i) = arma::sin(inputs_+phase(i));
    }
  }

  void Dataset::Generate ( const std::string& type,
                           const double& noise_std )
  {
    double scale = 1.; double phase = 0.;
    this -> Generate(scale, phase, type);
    this -> Noise(noise_std); 
  }

  void Dataset::Generate ( const size_t& M, const std::string& type,
                           const arma::rowvec& noise_std )
  {
    this -> Generate(M, type);
    this -> Noise(M, noise_std); 
  }

  void Dataset::Generate ( const size_t& M, const std::string& type,
                           const double& noise_std )
  {
    this -> Generate(M, type);
    this -> Noise(M, noise_std); 
  }

  void Dataset::Noise ( const double& noise_std )
  {
    arma::rowvec noise = arma::randn(1,size_)*noise_std;
    labels_.each_row() += noise;
  }

  void Dataset::Noise ( const size_t& M, const double& noise_std )
  {
    arma::rowvec noise = arma::randn(M,size_)*noise_std;
    labels_.each_row() += noise;
  }

  void Dataset::Noise ( const size_t& M, const arma::rowvec& noise_std )
  {
    arma::rowvec noise = arma::randn(M,size_)%noise_std;
    labels_.each_row() += noise;
  }


  void Dataset::Save( const std::string& filename )
  {
    arma::mat data = arma::join_cols(inputs_, labels_);
    utils::Save(filename, data, true);
  }

} // namespace regression
} // namespace data
} // namespace utils

namespace utils {
namespace data {
namespace regression {

  SineGen::SineGen ( ) { }

  SineGen::SineGen ( const size_t& M )
  {
    size_ = M;

    a_ = arma::rowvec(size_, arma::fill::randn);
    p_ = arma::rowvec(size_, arma::fill::randn);
  }

  arma::mat SineGen::Predict ( const arma::mat& inputs, 
                               const std::string& type,
                               const double& eps ) const 
  {
    arma::mat labels(size_, inputs.n_cols);
    if ( type == "Phase" )
    {
      for ( size_t i=0; i<size_; i++ )
        labels.row(i) = arma::sin(inputs+p_(i));
    }
    else if ( type == "Amplitude" )
    {
      for ( size_t i=0; i<size_; i++ )
        labels.row(i) = a_(i) * arma::sin(inputs);
    }
    else if ( type == "PhaseAmplitude" )
    {
      for ( size_t i=0; i<size_; i++ )
        labels.row(i) = a_(i)*arma::sin(inputs+p_(i));
    }

    if ( eps != 0. )
      labels += arma::randn(arma::size(labels), arma::distr_param(0.,eps));

    return labels;
  }
  arma::mat SineGen::Predict ( const arma::mat& inputs, 
                               const std::string& type ) const
  {
    return this-> Predict (inputs, type, 0.);
  }
} // namespace regression
} // namespace data
} // namespace utils

namespace utils {
namespace data {
namespace classification {

  Dataset::Dataset ( ) { }
  Dataset::Dataset ( const size_t& D, const size_t& N, const size_t& Nc ) :
                     size_(N), dimension_(D), num_class_(Nc) { }
  
  void Dataset::Generate ( const std::string& type )
  {
    if ( type  == "Dipping" )
    {
      BOOST_ASSERT_MSG( dimension_ == 1, "Dimension should be 1!" );
      BOOST_ASSERT_MSG( num_class_ == 2, "Class number should be 1!" );

      arma::mat i11, i12, i2;

      i11 = arma::randu<arma::mat>(1,size_/2, arma::distr_param(-2.5, -1.5));
      i12 = arma::randu<arma::mat>(1,size_/2, arma::distr_param(1.5, 2.5));
      i2  = arma::randu<arma::mat>(1,size_,   arma::distr_param(-0.5, 0.5));

            
      inputs_ = arma::join_rows(i11,i12,i2);

      arma::rowvec l1(size_), l2(size_);
      l1.zeros(); l2.ones();
      labels_ = arma::join_rows(l1,l2);

    }

    else if ( type  == "Simple" || type == "Hard" )
    {

      BOOST_ASSERT_MSG( num_class_ == 2, "Class number should be 1!" );
      BOOST_ASSERT_MSG( dimension_ == 1 || dimension_ == 2,
                                                 "Class number should be 1!" );
      arma::vec mean1;
      arma::vec mean2;
      double eps;
      if ( dimension_ == 1 )
      {
        mean1 = {-5};
        mean2 = {+5};
      }
      else 
      {
        mean1 = {-5,0};
        mean2 = {+5,0};
      }
      if ( type == "Hard" )
        eps = 2.;
      else
        eps = 0.1;
      _2classgauss(mean1, mean2, eps);
 
    }

    else if ( type  == "Delayed-Dipping" )
    {
      BOOST_ASSERT_MSG( num_class_ == 2, "Class number should be 1!" );

      double r = 1;
      double noise_std = 0.01;
      this -> _dipping(r, noise_std);

    }
  }

  void Dataset::_dipping( const double& r, const double& noise_std )
  {
    arma::mat x1(dimension_, size_, arma::fill::randn);
    x1.each_row() /= arma::sqrt(arma::sum(arma::pow(x1,2),0));
    if ( r != 1 )
      x1 *= r;

    arma::mat cov(dimension_, dimension_, arma::fill::eye);
    arma::vec mean(dimension_);
    mean.zeros(); cov *= 0.1;
    arma::mat x2 = arma::mvnrnd(mean, cov, size_);

    inputs_ = arma::join_rows(x1,x2);

    if ( noise_std > 0 )
      inputs_ += arma::randn(dimension_,2*size_,
                             arma::distr_param(0., noise_std));

    arma::rowvec l1(size_), l2(size_);
    l1.zeros(); l2.ones();
    labels_ = arma::join_rows(l1,l2);

  }
  void Dataset::_2classgauss ( const arma::vec& mean1, 
                               const arma::vec& mean2,
                               const double&  eps )
  {
    arma::mat x1;
    arma::mat x2;
        arma::mat cov(dimension_,dimension_, arma::fill::eye);
    cov *= eps;
    x1 = arma::mvnrnd(mean1, cov, size_);
    x2 = arma::mvnrnd(mean2, cov, size_);
    inputs_ = arma::join_rows(x1,x2);

    arma::rowvec l1(size_), l2(size_);
    l1.zeros(); l2.ones();
    labels_ = arma::join_rows(l1,l2);
  }
  void Dataset::Save( const std::string& filename )
  {
    arma::mat data = arma::join_cols(inputs_, labels_);
    utils::Save(filename, data, true);
  }



} // namespace classification
} // namespace data
} // namespace utils


#endif
