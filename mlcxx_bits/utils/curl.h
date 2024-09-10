/**
 * @file curl.h
 * @author Ozgur Taylan Turan
 *
 * This file is for curl related extra functions
 *
 */

namespace utils {
// Callback function to write data to a string
size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp)
{
  ((std::string*)userp)->append((char*)contents, size * nmemb);
  return size * nmemb;
}

} // namespace utils
