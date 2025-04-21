/**
 * @file datagen_impl.h
 * @author Ozgur Taylan Turan
 *
 * A simple toy data generation interface
 *
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
Dataset<T>::Dataset ( const size_t& D,
                      const size_t& N ) : size_(N), dimension_(D) 
{
  mean_ = arma::zeros<arma::Col<T>>(dimension_);
  cov_ = arma::eye<arma::Mat<T>>(dimension_,dimension_);
}
template<class T>
Dataset<T>::Dataset ( const size_t& D,
                      const size_t& N,
                      const arma::Col<T> mean,
                      const arma::Mat<T> cov ) : size_(N), dimension_(D),
                                          mean_(mean), cov_(cov)  { } 
template<class T>
void Dataset<T>::Generate ( const double& scale,
                            const double& phase,
                            const std::string& type )
{
  inputs_ = arma::mvnrnd(mean_,cov_,size_);
  /* inputs_ = arma::randu(dimension_,size_); */
  if (type == "Linear")
    labels_ = scale*(arma::ones<arma::Mat<T>>(dimension_)).t()*inputs_+phase;
  if (type == "-1/1")
    labels_ = arma::randi<arma::Row<DTYPE>>(size_,arma::distr_param(-1,1));
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
  /* arma::Row<T> noise2 = arma::randn<arma::Row<T>>(1,size_)*noise_std; */
  labels_.each_row() += noise;
  /* labels_.each_row() -= noise2; */
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
// classification::Dataset
//=============================================================================
template<class T>
Dataset<T>::Dataset ( const arma::Mat<T>& inputs,
                      const arma::Row<size_t>& labels ) :
                      size_(labels.n_elem), dimension_(inputs.n_rows),
                      num_class_(arma::unique(labels).eval().n_elem),
                      inputs_(inputs),
                      labels_(labels) { }
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

namespace oml
{
//=============================================================================
// Dataset
//=============================================================================
template<class LTYPE,class T>
Dataset<LTYPE,T>::Dataset( const size_t& id, const std::filesystem::path& path ) : 
  id_(id), path_(path)
{
  std::filesystem::create_directories(filepath_);
  std::filesystem::create_directories(metapath_);
  
  meta_url_ = "https://www.openml.org/api/v1/data/" 
                          + std::to_string(id);

  std::filesystem::create_directories(metapath_);

  metafile_ = metapath_/(std::to_string(id)+".meta");
  file_ = filepath_ / (std::to_string(id) + ".arff");

  if (!std::filesystem::exists(metafile_))
    this->_fetchmetadata();
  
  down_url_ = _getdownurl(_readmetadata());

  if (!std::filesystem::exists(file_))
  {
    this->_download();
    this->_fetchmetadata();
    this->_load();
  }
  else
  {
    WARN("Dataset " << id_ << " is already present.");
    this->_load();
  }
}

template<class LTYPE,class T>
void Dataset<LTYPE,T>::Update ( const arma::Mat<T>& inputs,
                                const arma::Row<LTYPE>& labels  )
{
  inputs_ = inputs; labels_ = labels; this->_update_info();
}

template<class LTYPE,class T>
Dataset<LTYPE,T>::Dataset( const size_t& id ) : 
  id_(id), path_(DATASET_PATH/"openml")
{
  std::filesystem::create_directories(filepath_);
  std::filesystem::create_directories(metapath_);
  
  meta_url_ = "https://www.openml.org/api/v1/data/" 
                          + std::to_string(id);

  std::filesystem::create_directories(metapath_);

  metafile_ = metapath_/(std::to_string(id)+".meta");
  file_ = filepath_ / (std::to_string(id) + ".arff");


  if (!std::filesystem::exists(metafile_))
    this->_fetchmetadata();
  
  down_url_ = _getdownurl(_readmetadata());

  if (!std::filesystem::exists(file_))
  {
    this->_download();
    this->_load();
  }
  else
  {
    WARN("Dataset " << id_ << " is already present.");
    this->_load();
  }
}

template<class LTYPE,class T>
bool Dataset<LTYPE,T>::_download( )
{
    CURL* curl;
    CURLcode res;
    // Initialize CURL
    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();

    // Open file to write the downloaded data
    FILE* fp = fopen(file_.c_str(), "wb");
    if (!fp) 
    {
      ERR("Could not open file for writing: " << file_);
      curl_easy_cleanup(curl);
      curl_global_cleanup();
      return false;
    }

    // Set CURL options
    LOG(down_url_.c_str());
    curl_easy_setopt(curl, CURLOPT_URL, down_url_.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);

    // Perform the request
    res = curl_easy_perform(curl);

    // Check for errors
    if(res != CURLE_OK)
    {
      ERR("curl_easy_perform() failed: "<< curl_easy_strerror(res));
      fclose(fp);
      curl_easy_cleanup(curl);
      curl_global_cleanup();
      return false;
    }

    // Cleanup
    fclose(fp);
    curl_easy_cleanup(curl);
    curl_global_cleanup();

    LOG("Dataset " << id_ << " downloaded to " << file_ << ".");
    return true;
} 

template<class LTYPE,class T>
std::string Dataset<LTYPE,T>::_fetchmetadata()
{
  
  // Function to fetch metadata from OpenMLdd
  CURL* curl;
  CURLcode res;
  std::string readBuffer;

  curl_global_init(CURL_GLOBAL_DEFAULT);
  curl = curl_easy_init();
  if(curl) 
  {
    curl_easy_setopt(curl, CURLOPT_URL, meta_url_.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, utils::WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
    res = curl_easy_perform(curl);
    if(res != CURLE_OK) 
      ERR("curl_easy_perform() failed: " << curl_easy_strerror(res)); 
    curl_easy_cleanup(curl);
  }
  curl_global_cleanup();


  if(res == CURLE_OK) 
  {
    // Save readBuffer to a text file
    std::ofstream outFile(metafile_);
    if (outFile.is_open())
    {
      outFile << readBuffer;
      outFile.close();
    }
    else
      ERR("Unable to open file for writing.");
  }
  else
      ERR("Not Saving metadata.");
  return readBuffer;
}

template<class LTYPE,class T>
std::string Dataset<LTYPE,T>::_readmetadata()
{
  std::ifstream infile(metafile_);
  if (!infile.is_open())
  {
    ERR("Unable to open file for reading: " << metafile_ );
    return "";
  }
  std::stringstream buffer;
  buffer << infile.rdbuf();
  infile.close();
  return buffer.str();
}

template<class LTYPE,class T>
std::string Dataset<LTYPE,T>::_gettargetname( const std::string& metadata )
{
  // Define a regular expression to find the <default_target_value> element
  std::regex re(R"(<oml:default_target_attribute>(.*?)</oml:default_target_attribute>)");
  std::smatch match;

  // Search for the pattern in the XML data
  if (regex_search(metadata, match, re) && match.size() > 1)
  {
    return match.str(1); // Return the matched content
  }
  else
  {
    WARN("Cannot find the target name using 'class' instead!!!");
    return "class";
  }
}

template<class LTYPE,class T>
std::string Dataset<LTYPE,T>::_getdownurl( const std::string& metadata )
{
  // Define a regular expression to find the <default_target_value> element
  std::regex re(R"(<oml:url>(.*?)</oml:url>)");

  std::smatch match;

  // Search for the pattern in the XML data
  if (regex_search(metadata, match, re) && match.size() > 1)
  {
    return match.str(1); // Return the matched content
  }
  else
  {
    WARN("Probably something went wrong with meta data fetch!!!");
    return "";
  }
}

template<class LTYPE,class T>
int Dataset<LTYPE,T>::_findlabel ( const std::string& targetname )
{
  std::ifstream file(file_);
  std::string line;
  int index = 0;

  if (!file.is_open()) 
  {
    ERR("Error opening file: " << file_ );
    return -1; // Error code for file opening failure
  }

  while (std::getline(file, line))
  {
    line.erase(0, line.find_first_not_of(" \t")); // Trim leading whitespace
    if (line.find("@ATTRIBUTE") == 0 || line.find("@attribute") == 0) 
    {
      size_t start = line.find(' ') + 1; // Skip "@attribute"
      // Find the first space/tab after the attribute name
      size_t end = line.find_first_of(" \t", start); 
      if (end == std::string::npos) 
          end = line.length();

     // Extract the attribute name
      std::string name = line.substr(start, end - start);

      if (name == targetname || name == "'"+targetname+"'") 
        return index; // Attribute found, return its index

      ++index; // Increment index for each attribute line
    }
  }
  return -1; // Attribute not found
}

template<class LTYPE,class T>
bool Dataset<LTYPE,T>::_iscateg(const arma::Row<T>& row)
{
  std::set<int> distinctValues;
  
  // Iterate over the array
  for (size_t i = 0; i < row.n_elem; ++i) 
  {
    T value = row(i);
    
    // Check if the value is a whole number
    if (value == static_cast<int>(value)) 
        distinctValues.insert(static_cast<int>(value));
    else 
        // If any value is not a whole number, it's not categorical
        return false;
  }

  // If we have only a few distinct values, consider it categorical
  return distinctValues.size() < row.n_elem;
}

template<class LTYPE,class T>
arma::Row<size_t> Dataset<LTYPE,T>::_convcateg(const arma::Row<T>& row)
{
  std::unordered_map<T, size_t> valueToIndex;
  size_t categoryIndex = 0;

  // Create a mapping of unique values to integer indices
  for (size_t i = 0; i < row.n_elem; ++i) 
  {
      T value = row(i);
      // If the value has not been seen before, assign a new category
      if (valueToIndex.find(value) == valueToIndex.end()) 
          valueToIndex[value] = categoryIndex++;
  }

  // Create a new Row<size_t> to store the mapped categorical values
  arma::Row<size_t> categoricalRow(row.n_elem);

  // Map each original value to its corresponding categorical index
  for (size_t i = 0; i < row.n_elem; ++i) 
      categoricalRow(i) = valueToIndex[row(i)];

  return categoricalRow;
}

template<class LTYPE, class T>
arma::Row<size_t> Dataset<LTYPE,T>::_procrow(const arma::Row<T>& row)
{
  if (_iscateg(row)) 
  {
    // Return the row as is if it's already categorical
    // Convert it to a size_t type (though it might not be necessary 
    // if it's already integers)
    arma::Row<size_t> categoricalRow(row.n_elem);
    for (size_t i = 0; i < row.n_elem; ++i) 
        categoricalRow(i) = static_cast<size_t>(row(i));
    return categoricalRow;
  } 
  else 
    // Convert real values into categorical values
    return _convcateg(row);
}

template<class LTYPE,class T>
void Dataset<LTYPE,T>::Save( const std::string& filename )
{
  std::ofstream file(filename, std::ios::binary);
  if (!file) 
    ERR("\rCannot open file for writing: " << filename << std::flush);

  cereal::BinaryOutputArchive archive(file);
  archive(cereal::make_nvp("Dataset", *this));  // Serialize the current object
  LOG("\rDataset object saved to " << filename << std::flush);

}

template<class LTYPE,class T>
std::shared_ptr<Dataset<LTYPE,T>> Dataset<LTYPE,T>::Load 
                                                ( const std::string& filename )
{
  std::ifstream file(filename, std::ios::binary);
  if (!file) 
  {
    ERR("\rError: Cannot open file for reading: " << filename);
    return nullptr;
  }
  cereal::BinaryInputArchive archive(file);
  auto dataset = std::make_shared<Dataset<LTYPE,T>>();
  archive(cereal::make_nvp("Dataset", *dataset));// Deserialize into a new object
  LOG("\rDataset loaded from " << filename);
  return dataset;
}

template<class LTYPE,class T>
void Dataset<LTYPE,T>::_load( )
{
  int idx = -1;
  arma::Mat<DTYPE> data;
  mlpack::data::DatasetInfo info;
  mlpack::data::Load(file_.c_str(), data, info);
  idx =_findlabel(_gettargetname(_readmetadata()));

  if (idx<0)
    throw std::runtime_error("Cannot find the label!");

  if constexpr (std::is_same<LTYPE, size_t>::value)
    labels_ = _procrow(data.row(idx));
  else
    labels_ = data.row(idx);
  /* labels_ = _procrow(data.row(idx)); */
  data.shed_row(idx);
  inputs_ = data;
  this->_update_info();
}

template<class LTYPE,class T>
void Dataset<LTYPE,T>::_update_info( )
{
  dimension_ = inputs_.n_rows;
  size_ = inputs_.n_cols;
  if (std::is_same<LTYPE,size_t>::value)
    num_class_ = (arma::unique(labels_).eval()).n_elem;
}

} // namespace oml

} // namespace data


#endif
