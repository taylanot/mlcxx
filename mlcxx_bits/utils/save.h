/**
 * @file save.h
 * @author Ozgur Taylan Turan
 *
 * Utility for saving data without hassle
 *
 * TODO: 
 * - Input jem::string too.
 * - Incorporate more filetypes...
 *
 *
 */

#ifndef SAVE_H
#define SAVE_H

//// standard
//#include <string>
//// jem
//#include <jem/base/String.h>
//// local
//#include "convert.h"

namespace utils {

  std::string Extension ( const std::string& filename )
  {
    return filename.substr(filename.find_last_of(".")+1);
  }

  template<class T>
  arma::mat Combine ( const T& data1,
                      const T& data2, bool column=true )
  {
    if (column)
    {
      return arma::join_cols(data1, data2);
    }
    else
    {
      return arma::join_cols(data1, data2);
    }
  }

  template<class T>
  void Save ( const std::string& filename,
              T& data,
              const bool transpose=true )
  {
    if (transpose)
    {
      arma::inplace_trans(data);
    }

    std:: string ext = Extension(filename);

    if (ext == "csv")
    {
      data.save(filename,arma::csv_ascii);
    }

  }

  template<class T>
  void Save ( const std::string& filename,
              T& data1,
              T& data2,
              const bool transpose=true )
  {
    T data = Combine(data1,data2);

    Save(filename, data, transpose);
  }

} // namespace utils

#endif
