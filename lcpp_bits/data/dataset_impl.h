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

namespace  data {

//-----------------------------------------------------------------------------
// Dataset
//-----------------------------------------------------------------------------
template<class LABEL,class T>
Dataset<LABEL,T>::Dataset ( size_t dim, size_t seed ) : 
  dimension_(dim),seed_(seed) {  };
//-----------------------------------------------------------------------------
// Dataset::Linear
//-----------------------------------------------------------------------------
template<class LABEL,class T>
void Dataset<LABEL,T>::Linear ( const size_t N, const T noise_std )
{
  mlpack::RandomSeed(seed_.value());
  inputs_ = arma::randn(dimension_,N);
  labels_ = arma::ones(1,dimension_) * inputs_ + 
    arma::randn(1,N,arma::distr_param(T(0),T(noise_std)));
  this->_update_info();
}
//-----------------------------------------------------------------------------
// Dataset::Sine
//-----------------------------------------------------------------------------
template<class LABEL,class T>
void Dataset<LABEL,T>::Sine ( const size_t N, const T noise_std )
{
  mlpack::RandomSeed(seed_.value());
  inputs_ = arma::randn(dimension_,N);
  labels_ = arma::sin(arma::ones(1,dimension_) * inputs_) + 
    arma::randn(1,N,arma::distr_param(T(0),T(noise_std)));
  this->_update_info();
}
//-----------------------------------------------------------------------------
// Dataset::Banana
//-----------------------------------------------------------------------------
template<class LABEL,class T>
void Dataset<LABEL,T>::Banana ( const size_t N, const T delta )
{
  if (dimension_ != 2)
    WARNING("Dataset::Banana requires dimension to be 2,\
            overwriting dimension_!");
  mlpack::RandomSeed(seed_.value());
  double r = 5.;
  double s = 1.0;
  arma::Mat<T> i1, i2, temp;

  temp = 0.125*M_PI + 1.25*M_PI*
              arma::randu<arma::Mat<T>>(1,N, arma::distr_param(0.,1.));

  i1 = arma::join_cols(r*arma::sin(temp),r*arma::cos(temp));
  i1 += s*arma::randu<arma::Mat<T>>(2,N, arma::distr_param(0.,1.));

  temp = 0.375*M_PI - 1.25*M_PI*
              arma::randu<arma::Mat<T>>(1,N, arma::distr_param(0.,1.));

  i2 = arma::join_cols(r*arma::sin(temp),r*arma::cos(temp));
  i2 += s*arma::randu<arma::Mat<T>>(2,N, arma::distr_param(0.,1.));
  i2 -= 0.75*r;

  i2 += delta;

  inputs_ = arma::join_rows(i1,i2);

  labels_.set_size(2*N);
  if constexpr ( std::is_same<LABEL,arma::Row<size_t>>::value )
    labels_.subvec(0, N-1).zeros(); 
  else if constexpr ( std::is_same<LABEL,arma::Row<int>>::value )
    labels_.subvec(0, N-1).fill(-1); 
  labels_.subvec(N, 2*N - 1).ones();

  this->_update_info();
}
//-----------------------------------------------------------------------------
// Dataset::Dipping
//-----------------------------------------------------------------------------
template<class LABEL,class T>
void Dataset<LABEL,T>::Dipping ( const size_t N, const T r, const T noise_std )

{
  arma::Mat<T> x1(dimension_, N, arma::fill::randn);
  x1.each_row() /= arma::sqrt(arma::sum(arma::pow(x1,2),0));

  if ( r != 1 )
    x1 *= r;

  arma::Mat<T> cov(dimension_, dimension_, arma::fill::eye);
  arma::Col<T> mean(dimension_);
  mean.zeros(); cov *= 0.1;
  arma::Mat<T> x2 = arma::mvnrnd(mean, cov, N);

  inputs_ = arma::join_rows(x1,x2);

  if ( noise_std > 0 )
    inputs_ += arma::randn<arma::Mat<T>>(dimension_,2*N,
                           arma::distr_param(0., noise_std));

  labels_.set_size(2*N);
  if constexpr ( std::is_same<LABEL,arma::Row<size_t>>::value )
    labels_.subvec(0, N-1).zeros(); 
  else if constexpr ( std::is_same<LABEL,arma::Row<int>>::value )
    labels_.subvec(0, N-1).fill(-1); 
  labels_.subvec(N, 2*N - 1).ones();

  this->_update_info();
}

//-----------------------------------------------------------------------------
// Dataset::Gaussian
//-----------------------------------------------------------------------------
template<class LABEL,class T>
void Dataset<LABEL,T>::Gaussian ( const size_t N,
                                  const arma::Row<T>& means )

{
  size_t n_class = means.n_elem;

  inputs_.set_size(dimension_, N*n_class);
  labels_.set_size(N*n_class);
  if constexpr ( std::is_same<LABEL,arma::Row<size_t>>::value )
      labels_.subvec(0, N-1).zeros(); 
  else if constexpr ( std::is_same<LABEL,arma::Row<int>>::value )
  {
    assert ( means.n_elem == 2 && 
        "To use Row<int> you need binary classification problem");
    labels_.subvec(0, N-1).fill(-1); 
  }
   
  for (size_t i = 0; i < n_class ; ++i)
  {
    inputs_.cols(i * N, (i + 1) * N - 1) = means(i) 
                                    + arma::randn<arma::Mat<T>>(dimension_, N);

    if (i>0)
      labels_.subvec(N, 2*N - 1).ones();
  }

  this->_update_info();
}
//-----------------------------------------------------------------------------
// Dataset::_update_info
//-----------------------------------------------------------------------------
template<class LABEL,class T>
void Dataset<LABEL,T>::_update_info(  )
{
  size_ = inputs_.n_cols;
  dimension_ = inputs_.n_rows;
  if constexpr ( std::is_same<LABEL,arma::Row<size_t>>::value )
    num_class_ = arma::unique(labels_).eval().n_elem;
  else if constexpr ( std::is_same<LABEL,arma::Row<int>>::value )
    num_class_ = 2;
}
} // namespace data

namespace data::oml
{
//=============================================================================
// Dataset
//-----------------------------------------------------------------------------
template<class LTYPE,class T>
Dataset<LTYPE,T>::Dataset( const size_t& id, const std::filesystem::path& path ) : 
  id_(id), path_(path)
{
  std::filesystem::create_directories(filepath_);
  std::filesystem::create_directories(metapath_);
  
  meta_url_ = "https://www.openml.org/api/v1/data/" 
                          + std::to_string(id);

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
    WARNING("Dataset " << id_ << " is already present.");
    this->_load();
  }
}
///////////////////////////////////////////////////////////////////////////////
template<class LTYPE,class T>
void Dataset<LTYPE,T>::Update ( const arma::Mat<T>& inputs,
                                const arma::Row<LTYPE>& labels  )
{
  inputs_ = inputs; labels_ = labels; this->_update_info();
}
///////////////////////////////////////////////////////////////////////////////
template<class LTYPE,class T>
Dataset<LTYPE,T>::Dataset( const size_t& id ) : 
  Dataset( id,DATASET_PATH/"openml") { }
///////////////////////////////////////////////////////////////////////////////
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
///////////////////////////////////////////////////////////////////////////////
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
///////////////////////////////////////////////////////////////////////////////
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
///////////////////////////////////////////////////////////////////////////////
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
    WARNING("Cannot find the target name using 'class' instead!!!");
    return "class";
  }
}
///////////////////////////////////////////////////////////////////////////////
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
    WARNING("Probably something went wrong with meta data fetch!!!");
    return "";
  }
}
///////////////////////////////////////////////////////////////////////////////
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
///////////////////////////////////////////////////////////////////////////////
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
///////////////////////////////////////////////////////////////////////////////
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
///////////////////////////////////////////////////////////////////////////////
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
///////////////////////////////////////////////////////////////////////////////
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
///////////////////////////////////////////////////////////////////////////////
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
///////////////////////////////////////////////////////////////////////////////
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
///////////////////////////////////////////////////////////////////////////////
template<class LTYPE,class T>
void Dataset<LTYPE,T>::_update_info( )
{
  dimension_ = inputs_.n_rows;
  size_ = inputs_.n_cols;
  if (std::is_same<LTYPE,size_t>::value)
    num_class_ = (arma::unique(labels_).eval()).n_elem;
}
///////////////////////////////////////////////////////////////////////////////
//-----------------------------------------------------------------------------
// Collect
//-----------------------------------------------------------------------------
template<class T>
Collect<T>::Collect( const size_t& id ) : Collect( id, DATASET_PATH )
{
  std::filesystem::create_directories(metapath_);
  std::filesystem::create_directories(filespath_);
};
///////////////////////////////////////////////////////////////////////////////
template<class T>
Collect<T>::Collect( const arma::Row<size_t>& ids ) : size_(ids.n_elem),
                                                      keys_(ids)
{
  std::filesystem::create_directories(metapath_);
  std::filesystem::create_directories(filespath_);
};
///////////////////////////////////////////////////////////////////////////////
template<class T>
Collect<T>::Collect( const size_t& id, const std::filesystem::path& path ) : 
  id_(id), path_(path)
{
  std::filesystem::create_directories(metapath_);
  std::filesystem::create_directories(filespath_);
  url_ = "https://www.openml.org/api/v1/json/study/" + std::to_string(id);
  keys_ = _getkeys();
  size_ = keys_.n_elem;
}
///////////////////////////////////////////////////////////////////////////////
template<class T>
Dataset<T> Collect<T>::GetNext ( )
{
  return Dataset<T>(keys_[counter_++],filespath_);
}
///////////////////////////////////////////////////////////////////////////////
template<class T>
Dataset<T> Collect<T>::GetID ( const size_t& id )
{
  if (arma::any(arma::find(keys_ == id)))
    return Dataset<T>(id);
  else
  {
    ERR("Collect: cannot find the dataset, giving you the next instead...");
    return GetNext();
  }
}
///////////////////////////////////////////////////////////////////////////////
template<class T>
arma::Row<size_t> Collect<T>::_getkeys()
{
  
  // Function to fetch metadata from OpenML
  CURL* curl;
  CURLcode res;
  std::string readBuffer;

  curl_global_init(CURL_GLOBAL_DEFAULT);
  curl = curl_easy_init();
  if(curl) 
  {
    curl_easy_setopt(curl, CURLOPT_URL, url_.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, utils::WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
    res = curl_easy_perform(curl);
    if(res != CURLE_OK) 
      ERR("Collect:curl_easy_perform() failed: " << curl_easy_strerror(res)); 
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
      ERR("Collect:Unable to open file for writing.");
  }
  else
      ERR("Collect:Not Saving metadata.");
  std::vector<size_t> dataIds;

    // Regex to match the array inside "data_id": [...]
    // I will use data_id because I want to control the splits...
    std::regex arrayRegex(R"("data_id"\s*:\s*\[([^\]]+)\])");
    std::smatch match;

    // If the regex finds a match for the "data_id" array
    if (std::regex_search(readBuffer, match, arrayRegex))
    {
      // The first captured group (the numbers inside the brackets)
      std::string dataIdArrayStr = match[1].str();

      // Regex to find individual numbers in the array
      std::regex numberRegex(R"(\d+)");
      auto numbersBegin = std::sregex_iterator(dataIdArrayStr.begin(),
                                               dataIdArrayStr.end(),
                                               numberRegex);
      auto numbersEnd = std::sregex_iterator();

      // Iterate over each match (number) and add it to the vector
      for (std::sregex_iterator i = numbersBegin; i != numbersEnd; ++i) 
      {
        int dataId = std::stoi((*i).str());
        dataIds.push_back(dataId);
      }
    }
  return arma::conv_to<arma::Row<size_t>>::from(dataIds);
}

} // namespace oml



#endif
