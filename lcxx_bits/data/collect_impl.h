/**
 * @file collect_impl.h
 * @author Ozgur Taylan Turan
 *
 * Collection of datasets from openml
 */

#ifndef COLLECT_IMPL_H
#define COLLECT_IMPL_H

namespace data {
namespace oml {

//-----------------------------------------------------------------------------
// Collect
//-----------------------------------------------------------------------------
template<class T>
Collect<T>::Collect( const size_t& id ) : Collect( id, DATASET_PATH ) { };
///////////////////////////////////////////////////////////////////////////////
template<class T>
Collect<T>::Collect( const arma::Row<size_t>& ids ) : keys_(ids),
                                                      size_(keys_.n_elem) { };
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
///////////////////////////////////////////////////////////////////////////////
} // namesapce oml
} // namesapce data

#endif


