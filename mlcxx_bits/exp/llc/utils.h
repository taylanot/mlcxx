/**
 * @file utils.h
 * @author Ozgur Taylan Turan
 *
 */

#ifndef LLC_UTILS_H 
#define LLC_UTILS_H

namespace experiments {
namespace llc {

//=============================================================================
// select : Select the relavent parts of the database
//=============================================================================
arma::uvec select(const arma::field<std::string> header,
                  const std::string which)
{
  std::map<std::string, size_t> counts;
  std::map<std::string, arma::uvec> ids;

  size_t count;
  for (auto it=conf::keys.begin(); it != conf::keys.end(); ++it)
  {
    count = 0;
    for (size_t j = 0; j < header.n_elem; j++)
    {
      bool check = header(j).find(*it) != std::string::npos;
      if (check)
        count++;
        
    }
    counts[*it] = count;
    arma::uvec temp_ids(count);
    count = 0;
    for (size_t j = 0; j < header.n_elem; j++)
    {
      bool check = header(j).find(*it) != std::string::npos;
      if (check)
        temp_ids(count++) = j;
    }
    ids[*it] = temp_ids;
  }
  return ids[which];
}

//=============================================================================
// select2 : Select the relavent parts of the database
//=============================================================================
arma::uvec select2(const arma::field<std::string> header,
                   const std::string which)
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

} //namespace llc
} //namespacee experiments


#endif
