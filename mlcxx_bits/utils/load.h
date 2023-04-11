/**
 * @file load.h
 * @author Ozgur Taylan Turan
 *
 * Utility for loading data without hassle
 *
 * TODO: 
 *
 * - Incorporate more filetypes...
 * - Faster loader with std::vector & ifstream
 *
 *
 */

#ifndef LOAD_H
#define LOAD_H

namespace utils {

//-----------------------------------------------------------------------
// DataAnalytics
//-----------------------------------------------------------------------
void DataAnalytics ( arma::mat& data,
                     const bool& transpose ) 
{
  if ( transpose )
    arma::inplace_trans(data);

  PRINT_VAR(arma::mean(data));
  PRINT_VAR(arma::median(data));
  PRINT_VAR(arma::stddev(data));
  PRINT_VAR(arma::cor(data));
  PRINT_VAR(arma::max(data));
  PRINT_VAR(arma::min(data));
}

//-----------------------------------------------------------------------
// read_data
//-----------------------------------------------------------------------
arma::mat read_data ( const std::string& filename,
                      const bool& transpose)
{
  arma::mat matrix;
  mlpack::data::DatasetInfo info;
  mlpack::data::Load(filename,matrix,info,true,transpose);
  return matrix;
}

//-----------------------------------------------------------------------
// loadCSV
//-----------------------------------------------------------------------
arma::mat loadCSV ( const std::string &filename, 
                    const std::string &delimeter = "," )
{
  std::ifstream csv(filename);
  std::vector<std::vector<double>> datas;

  for ( std::string line; std::getline(csv, line); )
  {
      std::vector<double> data;

      // split string by delimeter
      auto start = 0U;
      auto end = line.find(delimeter);
      while (end != std::string::npos) {
          data.push_back(std::stod(line.substr(start, end - start)));
          start = end + delimeter.length();
          end = line.find(delimeter, start);
      }
      data.push_back(std::stod(line.substr(start, end)));
      datas.push_back(data);
  }

  arma::mat data_mat = arma::zeros<arma::mat>(datas.size(), datas[0].size());

  for (size_t i=0; i<datas.size(); i++) {
      arma::mat r(datas[i]);
      data_mat.row(i) = r.t();
  }

  return data_mat;
}
  arma::mat Load( const std::string& filename,
                  const bool transpose=false )
  {
    std:: string ext = Extension(filename);
    arma::mat data;

    if (ext == "csv")
    {
      //data.load(filename,arma::csv_ascii);
    }

    if (transpose)
    {
      arma::inplace_trans(data);
    }

    return data;
  }

} // namespace utils

#endif
