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
#ifndef DATASET_IMPL_H
#define DATASET_IMPL_H

namespace data {
namespace regression {

//=============================================================================
// Dataset
//=============================================================================
template<class T>
Dataset<T>::Dataset ( ) { }
template<class T>
Dataset<T>::Dataset ( const size_t& D,
                      const size_t& N ) : size_(N), dimension_(D) { }

template<class T>
void Dataset<T>::Generate ( const double& scale,
                            const double& phase,
                            const std::string& type )
{
  inputs_ = arma::mvnrnd(arma::zeros<arma::Col<T>>(dimension_),
                          arma::eye<arma::Mat<T>>(dimension_,dimension_),size_);
  if (type == "Linear")
    labels_ = scale*(arma::ones<arma::Mat<T>>(dimension_)).t()*inputs_+phase;
  else if (type == "RandomLinear")
  {
    auto a=arma::randn<arma::Col<T>>(dimension_,
                                    arma::distr_param(0., 1.));
    labels_ = scale*a.t()*inputs_+phase;
  }
  else if (type == "Sine")
    labels_ = scale*(arma::ones<arma::Mat<T>>(dimension_)).t()
                                                    * arma::sin(inputs_+phase);
  else if (type == "Sinc")
    labels_ = scale * (arma::ones<arma::Mat<T>>(dimension_)).t() * 
                          (arma::sin(inputs_ + phase) / inputs_);
}

template<class T>
void Dataset<T>::Generate ( const arma::Row<T>& a,
                            const double& noise_std )
{
  inputs_ = arma::mvnrnd(arma::zeros<arma::Col<T>>(dimension_),
                          arma::eye<arma::Mat<T>>(dimension_,dimension_),size_);
  labels_ = a*inputs_;
  this -> Noise(noise_std);
}

template<class T>
void Dataset<T>::Generate ( const double& scale,
                            const double& phase,
                            const std::string& type,
                            const double& noise_std )
{
    this -> Generate(scale, phase, type);
    this -> Noise(noise_std);
}


template<class T>
void Dataset<T>::Generate ( const std::string& type,
                            const double& noise_std )
{
  double scale = 1.; double phase = 0.;
  this -> Generate(scale, phase, type);
  this -> Noise(noise_std); 
}

template<class T>
void Dataset<T>::Noise ( const double& noise_std )
{
  arma::Row<T> noise = arma::randn<arma::Row<T>>(1,size_)*noise_std;
  labels_.each_row() += noise;
}


template<class T>
void Dataset<T>::Save( const std::string& filename )
{
  arma::Mat<T> data = arma::join_cols(inputs_, labels_);
  utils::Save(filename, data, true);
}

template<class T>
void Dataset<T>::Load ( const std::string& filename,
                        const size_t& Din,
                        const size_t& Dout,
                        const bool transpose,
                        const bool count )
{
  arma::Mat<T> data = utils::Load(filename, transpose, count);
  BOOST_ASSERT_MSG( Din+Dout == data.n_rows, "Dimension does not match!" );
  size_ = data.n_cols;
  dimension_ = Din;
  inputs_ = data.rows(0,dimension_-1);
  labels_ = data.row(dimension_);
}

template<class T>
void Dataset<T>::Outlier_( const size_t& Nout )
{
    BOOST_ASSERT_MSG( dimension_ == 1,
        "Only 1 dimensional Outlier dataset is defined!" );
    BOOST_ASSERT_MSG( size_ > Nout,
        "Number of outliers is bigger than dataset size" );
    arma::Mat<T> inp1 = arma::randn<arma::Mat<T>>(dimension_,size_t(size_/2)
                                                              -size_t(Nout/2));
    inp1 +=1;
    arma::Mat<T> inp2 = arma::randn<arma::Mat<T>>(dimension_,size_t(size_/2)
                                                              -size_t(Nout/2));
    inp2 -=1;
    arma::Mat<T> out1 = arma::zeros<arma::Mat<T>>(dimension_,size_t(Nout/2));
    out1 -= 1000;
    arma::Mat<T> out2 = arma::zeros<arma::Mat<T>>(dimension_,size_t(Nout/2));
    out2 += 1000;
    arma::Mat<T> lab1 = arma::ones<arma::Mat<T>>(dimension_,size_t(size_/2));
    arma::Mat<T> lab2 = arma::ones<arma::Mat<T>>(dimension_,size_t(size_/2));
    lab2 -= 2;
    inputs_ = arma::join_horiz(inp1,out1,inp2,out2);
    labels_ = arma::join_horiz(lab1,lab2);

}

template<class T>
void Dataset<T>::Generate( const std::string& type )
{
  if (type == "Outlier-1")
    Dataset::Outlier_(1);

  else if (type == "Outlier-10")
    Dataset::Outlier_(10);

  else if (type == "Linear" || type == "Sine" ||
           type == "Sinc" || type == "RandomLinear" )
    Dataset::Generate(1,0,type);
}

template<class T>
void Dataset<T>::Load ( const std::string& filename,
                        const arma::uvec& ins, 
                        const arma::uvec& outs,
                        const bool transpose,
                        const bool count )
{
  arma::Mat<T> data = utils::Load(filename, transpose, count);
  BOOST_ASSERT_MSG( ins.n_elem+outs.n_elem== data.n_rows, "Dimension does not match!" );
  size_ = data.n_cols;
  dimension_ = ins.n_elem;
  inputs_ = data.rows(ins);
  labels_ = data.rows(outs);
}

template<class T>
arma::Row<T> Dataset<T>::GetLabels ( const size_t id )
{
  return labels_.row(id);
}

} // namespace regression
} // namespace data

namespace data {
namespace functional {

template<class T>
Dataset<T>::Dataset ( ) { }

template<class T>
Dataset<T>::Dataset ( const size_t& D,
                      const size_t& N,
                      const size_t& M ) : size_(N), dimension_(D), nfuncs_(M)
                      { }

template<class T>
void Dataset<T>::Generate ( const std::string& type )
{
  labels_.resize(nfuncs_,size_);  

  if (type == "Sine")
  {
    inputs_ = arma::sort(arma::randn<arma::Mat<T>>(dimension_,size_),"ascend",1);
    arma::Row<T> phase(nfuncs_,arma::fill::randn);
    for( size_t i=0; i<nfuncs_; i++ )
       labels_.row(i) = arma::sin(inputs_+phase(i));
  }
}

template<class T>
void Dataset<T>::Generate ( const std::string& type,
                            const arma::Row<T>& noise_std )
{
  this -> Generate(type);
  this -> Noise(noise_std); 
}

template<class T>
void Dataset<T>::Generate ( const std::string& type,
                            const double& noise_std )
{
  this -> Generate(type);
  this -> Noise(noise_std); 
}

template<class T>
void Dataset<T>::Noise ( const double& noise_std )
{
  arma::Row<T> noise = arma::randn<arma::Row<T>>(1,size_)*noise_std;
  labels_.each_row() += noise;
}

template<class T>
void Dataset<T>::Noise (const arma::Row<T>& noise_std )
{
  arma::Row<T> noise = arma::randn<arma::Row<T>>(nfuncs_,size_)%noise_std;
  labels_.each_row() += noise;
}


template<class T>
void Dataset<T>::Save( const std::string& filename )
{
  arma::Mat<T> data = arma::join_cols(inputs_, labels_);
  utils::Save(filename, data, true);
}

template<class T>
void Dataset<T>::Load ( const std::string& filename,
                        const size_t& Din,
                        const size_t& Dout,
                        const bool& transpose,
                        const bool& count )
{
  arma::Mat<T> data = utils::Load(filename, transpose, count);
  BOOST_ASSERT_MSG( Din+Dout == data.n_rows, "Dimension does not match!" );
  size_ = data.n_cols;
  dimension_ = Din;
  inputs_ = data.rows(0,dimension_-1);
  labels_ = data.row(dimension_);
}

template<class T>
void Dataset<T>::Normalize ( )
{
  arma::Row<T> variance = arma::var(labels_,0);
  weights_ = arma::trapz(inputs_,variance,1);
  labels_ = labels_/weights_(0,0);
}

template<class T>
void Dataset<T>::UnNormalize ( )
{
  labels_ = labels_*weights_(0,0);
}
//=============================================================================
// SineGen
//=============================================================================

template<class T>
SineGen<T>::SineGen ( ) { }

template<class T>
SineGen<T>::SineGen ( const size_t& M )
{
  size_ = M;

  a_ = arma::Row<T>(size_, arma::fill::randn);
  p_ = arma::Row<T>(size_, arma::fill::randn);
}

template<class T>
arma::Mat<T> SineGen<T>::Predict ( const arma::Mat<T>& inputs, 
                                   const std::string& type,
                                   const double& eps ) const 
{
  arma::Mat<T> labels(size_, inputs.n_cols);
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
    labels += arma::randn<arma::Mat<T>>(arma::size(labels), arma::distr_param(0.,eps));

  return labels;
}

template<class T>
arma::Mat<T> SineGen<T>::Mean ( const arma::Mat<T>& inputs, 
                                const std::string& type,
                                const double& eps ) const 
{
  auto labels = Predict(inputs, type, eps);
  return arma::zeros(arma::size(arma::mean(labels, 0)));
}

template<class T>
arma::Mat<T> SineGen<T>::Mean ( const arma::Mat<T>& inputs, 
                                const std::string& type ) const 
{
  auto labels = Predict(inputs, type);

  return arma::zeros(arma::size(arma::mean(labels, 0)));
}

template<class T>
arma::Mat<T> SineGen<T>::Predict ( const arma::Mat<T>& inputs, 
                                   const std::string& type ) const
{
  return this-> Predict (inputs, type, 0.);
}

template<class T>
size_t SineGen<T>::GetM ( )
{
  return size_;
}

} // namespace functional
} // namespace data

namespace data {
namespace classification {

//=============================================================================
// Dataset
//=============================================================================

template<class T>
Dataset<T>::Dataset ( ) { }

template<class T>
Dataset<T>::Dataset ( const size_t& D,
                      const size_t& N,
                      const size_t& Nc ) :
                      size_(N), dimension_(D), num_class_(Nc) { }

template<class T>
void Dataset<T>::Generate ( const std::string& type )
{
  if ( type  == "Banana" )
  {
    BOOST_ASSERT_MSG( dimension_ == 2, "Dimension should be 1!" );
    BOOST_ASSERT_MSG( num_class_ == 2, "Class number should be 2!" );
     
    this -> _banana(0.);
    
  }

  else if ( type  == "Dipping" )
  {
    BOOST_ASSERT_MSG( dimension_ == 1, "Dimension should be 1!" );
    BOOST_ASSERT_MSG( num_class_ == 2, "Class number should be 2!" );

    arma::Mat<T> i11, i12, i2;

    i11 = arma::randu<arma::Mat<T>>(1,size_/2, arma::distr_param(-2.5, -1.5));
    i12 = arma::randu<arma::Mat<T>>(1,size_/2, arma::distr_param(1.5, 2.5));
    i2  = arma::randu<arma::Mat<T>>(1,size_,   arma::distr_param(-0.5, 0.5));

          
    inputs_ = arma::join_rows(i11,i12,i2);

    arma::Row<size_t> l1(size_), l2(size_);
    l1.zeros(); l2.ones();
    labels_ = arma::join_rows(l1,l2);

  }

  else if ( type  == "Simple" || type == "Hard" || type == "Harder" )
  {

    BOOST_ASSERT_MSG( num_class_ == 2, "Class number should be 2!" );
    BOOST_ASSERT_MSG( dimension_ == 1 || dimension_ == 2,
                                               "Dimension should be 1!" );
    arma::Col<T> mean1;
    arma::Col<T> mean2;
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
      eps = 4.;
    else if ( type == "Harder" )
      eps = 5.;
    else
      eps = 0.1;
    _2classgauss(mean1, mean2, eps, 0.);

  }

  else if ( type  == "Delayed-Dipping" )
  {
    BOOST_ASSERT_MSG( num_class_ == 2, "Class number should be 1!" );

    double r = 20.;
    double noise_std = 1.;
    this -> _dipping(r, noise_std);

  }
}

template<class T>
void Dataset<T>::_banana ( const double& delta )
{

  double r = 5.;
  double s = 1.0;
  arma::Mat<T> i1, i2, temp;

  temp = 0.125*M_PI + 1.25*M_PI*
              arma::randu<arma::Mat<T>>(1,size_, arma::distr_param(0.,1.));

  i1 = arma::join_cols(r*arma::sin(temp),r*arma::cos(temp));
  i1 += s*arma::randu<arma::Mat<T>>(2,size_, arma::distr_param(0.,1.));

  temp = 0.375*M_PI - 1.25*M_PI*
              arma::randu<arma::Mat<T>>(1,size_, arma::distr_param(0.,1.));

  i2 = arma::join_cols(r*arma::sin(temp),r*arma::cos(temp));
  i2 += s*arma::randu<arma::Mat<T>>(2,size_, arma::distr_param(0.,1.));
  i2 -= 0.75*r;

  i2 += delta;

  inputs_ = arma::join_rows(i1,i2);

  arma::Row<size_t> l1(size_), l2(size_);
  l1.zeros(); l2.ones();
  labels_ = arma::join_rows(l1,l2);
}


template<class T>
void Dataset<T>::_dipping ( const double& r, const double& noise_std )
{
  arma::Mat<T> x1(dimension_, size_, arma::fill::randn);
  x1.each_row() /= arma::sqrt(arma::sum(arma::pow(x1,2),0));
  if ( r != 1 )
    x1 *= r;

  arma::Mat<T> cov(dimension_, dimension_, arma::fill::eye);
  arma::Col<T> mean(dimension_);
  mean.zeros(); cov *= 0.1;
  arma::Mat<T> x2 = arma::mvnrnd(mean, cov, size_);

  inputs_ = arma::join_rows(x1,x2);

  if ( noise_std > 0 )
    inputs_ += arma::randn<arma::Mat<T>>(dimension_,2*size_,
                           arma::distr_param(0., noise_std));

  arma::Row<size_t> l1(size_), l2(size_);
  l1.zeros(); l2.ones();
  labels_ = arma::join_rows(l1,l2);

}

template<class T>
void Dataset<T>::_2classgauss ( const arma::Col<T>& mean1, 
                                const arma::Col<T>& mean2,
                                const double& eps,
                                const double& delta )
{
  arma::Mat<T> x1;
  arma::Mat<T> x2;
  arma::Mat<T> cov(dimension_,dimension_, arma::fill::eye);
  cov *= eps;
  x1 = arma::mvnrnd(mean1, cov, size_);
  x2 = arma::mvnrnd(mean2, cov, size_);
  x2.row(0) += delta;
  inputs_ = arma::join_rows(x1,x2);

  arma::Row<size_t> l1(size_), l2(size_);
  l1.zeros(); l2.ones();
  labels_ = arma::join_rows(l1,l2);
}

template<class T>
void Dataset<T>::_imbalance2classgauss ( const double& perc )
{
  arma::Mat<T> x1;
  arma::Mat<T> x2;

  arma::Mat<T> cov(dimension_,dimension_, arma::fill::eye);
  
  size_t size1 = std::floor(2*size_*perc);
  size_t size2 = 2 * size_ - size1;

  arma::Col<T> mean1 = {0,0};
  arma::Col<T> mean2 = {1,0};

  x1 = arma::mvnrnd(mean1, cov, size1);
  x2 = arma::mvnrnd(mean2, cov, size2);

  inputs_ = arma::join_rows(x1,x2);

  arma::Row<size_t> l1(size1), l2(size2);
  l1.zeros(); l2.ones();
  labels_ = arma::join_rows(l1,l2);
}

template<class T>
void Dataset<T>::Save( const std::string& filename )
{
  arma::Row<T> type_match = arma::conv_to<arma::Row<T>>::from(labels_);
  arma::Mat<T> data = arma::join_cols(inputs_, type_match);
  utils::Save(filename, data, true);
}

template<class T>
void Dataset<T>::Load ( const std::string& filename,
                        const bool last,
                        const bool transpose,
                        const bool count )
{
  arma::Mat<T> data = utils::Load(filename,transpose,count);
  size_ = data.n_cols;
  dimension_ = data.n_rows-1;
  num_class_ = arma::max(data.row(dimension_))+1;
  if (last)
  {
    inputs_ = data.rows(0,dimension_-1);
    labels_ = arma::conv_to<arma::Row<size_t>>::from(data.row(dimension_));
  }
  else
  {
    inputs_ = data.rows(1,dimension_-1);
    labels_ = arma::conv_to<arma::Row<size_t>>::from(data.row(0));
  }
}

} // namespace classification
} // namespace data


#endif
