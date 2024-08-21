/**
 * @file nodd.cpp
 * @author Ozgur Taylan Turan
 *
 * This file is for showing double descent mitigation via excluisoin of principle components.
 *
 */
#define DTYPE double  

#include <headers.h>
// Function to download a dataset from OpenML using its data ID
bool getopenmldataset( int data_id )
{
  CURL* curl;
  CURLcode res;

  // Construct the URL for the dataset
  std::string url = "https://www.openml.org/data/download/" 
          + std::to_string(data_id);
  std::filesystem::path path = DATASET_PATH / "openml";
  std::filesystem::create_directories(path);
  std::filesystem::path filename = DATASET_PATH / "openml" /(std::to_string(data_id) + ".arff");

  // Initialize CURL
  curl_global_init(CURL_GLOBAL_DEFAULT);
  curl = curl_easy_init();

  // Open file to write the downloaded data
  FILE* fp = fopen(filename.c_str(), "wb");
  if (!fp) 
  {
    std::cerr << "Could not open file for writing: " << filename << std::endl;
    curl_easy_cleanup(curl);
    curl_global_cleanup();
    return false;
  }

  // Set CURL options
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);

  // Perform the request
  res = curl_easy_perform(curl);

  // Check for errors
  if(res != CURLE_OK)
  {
    std::cerr << "\n" << "curl_easy_perform() failed: "
      << curl_easy_strerror(res) << std::endl;

    fclose(fp);
    curl_easy_cleanup(curl);
    curl_global_cleanup();
    return false;
  }

  // Cleanup
  fclose(fp);
  curl_easy_cleanup(curl);
  curl_global_cleanup();

  std::cout << "Dataset " << data_id << " downloaded to " 
    << filename << "." << std::endl;

  return true;
}

int main ( int argc, char** argv )
{
  getopenmldataset(61);  
  return 0;
}
