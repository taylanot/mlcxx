/**
 * @file data_mani.h
 * @author Ozgur Taylan Turan
 *
 * Data manipulation related stuff
 *
 * TODO: 
 *
 *
 */

#ifndef SYSTEM_H
#define SYSTEM_H

namespace utils {

//=============================================================================
// select_header : Select the relavent parts of the database by using 
//                  arma::field
//=============================================================================
arma::uvec select_header(const arma::field<std::string>& header,
                         const std::string& which)
{

  size_t count=0;
  for (size_t j = 0; j < header.n_elem; j++)
  {
    bool check = header(j).find(which) != std::string::npos;
    if (check)
      count++;
  }

  arma::uvec ids(count);
  count = 0;
  for (size_t j = 0; j < header.n_elem; j++)
  {
    bool check = header(j).find(which) != std::string::npos;
    if (check)
      ids(count++) = j;
  }
  return ids;
}

//=============================================================================
// select_header : Selecting multiple relavent parts and finding the 
//                  intersection between those by using arma::field
//=============================================================================
arma::uvec select_headers(const arma::field<std::string>& header,
                          const std::vector<std::string>& whichs)
{
  BOOST_ASSERT_MSG ( whichs.size() > 1,
      "The size of the search vector should be more than 1!");

  arma::field<arma::uvec> ids(1,whichs.size());

  size_t counter = 0;
  for (std::string which: whichs)
    ids(0,counter++) = select_header(header,which);

  arma::uvec id;
  for (size_t i=0; i<whichs.size()-1;i++)
  {
    if (i ==0)
      id = arma::intersect(ids(i),ids(i+1));
    else
      id = arma::intersect(id,ids(i+1));

  }


  return id;
}

} // namespace utils

#endif
