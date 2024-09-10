/**
 * @file nodd.cpp
 * @author Ozgur Taylan Turan
 *
 * This file is for showing double descent mitigation via excluisoin of principle components.
 *
 */
#define DTYPE double  

#include <headers.h>

int findLabelARFF (const std::string& filename, const std::string& attrname)
{
  std::ifstream file(filename);
  std::string line;
  int index = 0;

  if (!file.is_open()) 
  {
    std::cerr << "Error opening file: " << filename << std::endl;
    return -1; // Error code for file opening failure
  }

  while (std::getline(file, line))
  {
    line.erase(0, line.find_first_not_of(" \t")); // Trim leading whitespace
    if (line.find("@ATTRIBUTE") == 0 || line.find("@attribute" == 0)) 
    {
      size_t start = line.find(' ') + 1; // Skip "@attribute"
      // Find the first space/tab after the attribute name
      size_t end = line.find_first_of(" \t", start); 
      if (end == std::string::npos) 
          end = line.length();

      std::string name = line.substr(start, end - start); // Extract the attribute name
      PRINT(name);

      if (name == attrname || name == "'"+attrname+"'") 
        return index; // Attribute found, return its index

      ++index; // Increment index for each attribute line
    }
  }
  return -1; // Attribute not found
}

// Function to download a dataset from OpenML using its data ID
bool getopenmldataset( size_t id )
{
    // Construct the URL for the dataset
  std::string url = "https://www.openml.org/data/download/" 
          + std::to_string(id);
  std::filesystem::path path = DATASET_PATH / "openml";
  std::filesystem::create_directories(path);
  std::filesystem::path filename = DATASET_PATH / "openml" /(std::to_string(id) + ".arff");

  if (!std::filesystem::exists(filename))
  {
    CURL* curl;
    CURLcode res;


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

    std::cout << "Dataset " << id << " downloaded to " 
      << filename << "." << std::endl;

    return true;
  }
  else
  {
    std::cout <<"Dataset "<< id <<" is already downloaded."<< std::endl;
    return true;
  }
}

// Function to extract default_target_value from XML data using regex
std::string extractDefaultTargetValue(const std::string& xmlData)
{
  // Define a regular expression to find the <default_target_value> element
  std::regex re(R"(<oml:default_target_attribute>(.*?)</oml:default_target_attribute>)");
  std::smatch match;

  // Search for the pattern in the XML data
  if (regex_search(xmlData, match, re) && match.size() > 1) {
      return match.str(1); // Return the matched content
  }
  return "";
}

size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp)
{
  ((std::string*)userp)->append((char*)contents, size * nmemb);
  return size * nmemb;
}

// Function to fetch metadata from OpenML
std::string fetchMetadata(size_t id)
{
  
  std::string url = "https://www.openml.org/api/v1/data/" + std::to_string(id);
  CURL* curl;
  CURLcode res;
  std::string readBuffer;

  curl_global_init(CURL_GLOBAL_DEFAULT);
  curl = curl_easy_init();
  if(curl) 
  {
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, utils::WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
    res = curl_easy_perform(curl);
    if(res != CURLE_OK) 
      std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
    curl_easy_cleanup(curl);
  }
  curl_global_cleanup();

  return readBuffer;
}

int main ( int argc, char** argv )
{
  std::filesystem::path path = "30_08_24/oml/3";
  /* std::filesystem::create_directories(path); */
  /* std::filesystem::current_path(path); */

  arma::wall_clock timer;
  timer.tic();


  /* size_t id = 37; // iris dataset */
  /* getopenmldataset(id); */  
  /* arma::Mat<DTYPE> data; */
  /* std::filesystem::path file = DATASET_PATH/"openml"/(std::to_string(id)+".arff"); */
  /* mlpack::data::DatasetInfo info; */
  /* mlpack::data::Load(file.c_str(), data, info); */
  /* PRINT(data.size()); */

  /* PRINT(extractDefaultTargetValue(xmlData)); */

  /* PRINT(fetchMetadata(61)); */

  /* PRINT(findLabelARFF(file.c_str(),"'class'")); */
  /* PRINT(findLabelARFF(file.c_str(),extractDefaultTargetValue(fetchMetadata(id)))); */
  /* PRINT(extractDefaultTargetValue(fetchMetadata(id))); */
  /* mlpack::data::LoadARFF<DTYPE>(file,mat); */

  /* data::classification::oml::Dataset dataset(11); */
  data::classification::oml::Dataset dataset(3);

  data::classification::oml::Dataset trainset,testset;
  data::StratifiedSplit(dataset,trainset,testset,0.2);

  /* data::classification::Dataset trainset(2,100,2); */
  /* data::classification::Dataset testset(2,1000,2); */
  /* trainset.Generate("Simple"); */
  /* testset.Generate("Simple"); */
  

  size_t repeat = 10;
  /* arma::irowvec Ns = arma::regspace<arma::irowvec>(2,1,size_t(trainset.size_*0.3)); */
  arma::irowvec Ns = arma::regspace<arma::irowvec>(2,100,2500);
  /* src::LCurve<mlpack::LinearSVM<>,mlpack::Accuracy> lcurve(Ns,repeat,true,false,true); */
  /* src::LCurve<algo::classification::KernelSVM<>,mlpack::Accuracy> lcurve(Ns,repeat,true,false,true); */
  /* src::LCurve<algo::classification::KernelSVM<mlpack::GaussianKernel>, */
  /* src::LCurve<algo::classification::LDC<>, */
  /* src::LCurve<algo::classification::KernelSVM<mlpack::LinearKernel>, */
  /* src::LCurve<algo::classification::QDC<>, */
              /* mlpack::Accuracy> lcurve(Ns,repeat,true,false,true); */
  src::LCurve<algo::classification::KernelSVM<mlpack::GaussianKernel>,mlpack::Accuracy,
    data::N_StratSplit> lcurve(Ns,repeat,true,false,true);
  /* src::LCurve<algo::classification::QDC<>,mlpack::Accuracy> lcurve(Ns,repeat,true,false,true); */
  /* src::LCurve<algo::classification::LDC<>,mlpack::Accuracy> lcurve(Ns,repeat,true,false,true); */
  /* lcurve.Split(trainset,testset,2,1.e-6); */
  /* lcurve.Split(trainset,testset,10.); */
  lcurve.Split(trainset,testset);
  PRINT(lcurve.test_errors_.save("svc.csv",arma::csv_ascii));

  PRINT_TIME(timer.toc());

  
  return 0;
}
