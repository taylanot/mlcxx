/**
 * @file sandl.h
 * @author Ozgur Taylan Turan
 *
 * Utility for saving data without hassle
 *
 * TODO: Save Catagoriacal
 *
 *
 */

#ifndef SANDL_H
#define SANDL_H

namespace utils {

//-----------------------------------------------------------------------------
// CreateDirs: Create direcotries if the path does not exits.
//-----------------------------------------------------------------------------
void CreateDirs ( const std::filesystem::path& filename )
{
  bool has_path = filename.has_parent_path();
  if (has_path)
    std::filesystem::create_directories(filename.parent_path());
}
//-----------------------------------------------------------------------------
// Extension  : Find the extension of a filename  
//-----------------------------------------------------------------------------
std::string Extension ( const std::string& filename )
{
  return filename.substr(filename.find_last_of(".")+1);
}

//-----------------------------------------------------------------------------
// Combine  : Combine two matricies
//-----------------------------------------------------------------------------
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

//-----------------------------------------------------------------------------
// Save : Save a data container to a file
//-----------------------------------------------------------------------------
template<class T>
void Save ( const std::filesystem::path& filename,
            const T& data,
            const bool transpose=true )
{
  T temp;

  if (transpose)
    temp = data.t();
  else
    temp = data;

  std::string ext = Extension(filename);

  CreateDirs(filename);

  if (ext == "csv")
  {
    temp.save(filename,arma::csv_ascii);
  }
  else if (ext == "bin")
  {
    temp.save(filename,arma::arma_binary);
  }
  else if (ext == "arma")
  {
    temp.save(filename,arma::arma_ascii);
  }
  else if (ext == "txt")
  {
    temp.save(filename,arma::raw_ascii);
  }
  else
    throw std::runtime_error("Not Implemented save extension!");

}

//-----------------------------------------------------------------------------
// Save : Save two data containers to a file
//-----------------------------------------------------------------------------
template<class T>
void Save ( const std::filesystem::path& filename,
            T& data1,
            T& data2,
            const bool transpose=true )
{
  T data = Combine(data1,data2);

  Save(filename, data, transpose);
}

//-----------------------------------------------------------------------
// Load
//-----------------------------------------------------------------------
template<class T>
arma::mat Load ( const T& filename,
                 const bool& transpose,
                 const bool& count = false )
{
  arma::mat matrix;
  mlpack::data::DatasetInfo info;
  if ( count )
  {
    mlpack::data::Load(filename,matrix,info,true,transpose);
  }
  else
    mlpack::data::Load(filename,matrix,true,transpose);
  return matrix;
}

//-----------------------------------------------------------------------
// LoadwHeader
//-----------------------------------------------------------------------
template<class T>
std::tuple<arma::mat,arma::field<std::string>> 
                      LoadwHeader ( const T& filename,
                                    const bool& transpose )
{
  arma::mat matrix;
  arma::field<std::string> header;
  matrix.load(arma::csv_name(filename, header));
  if (transpose)
    arma::inplace_trans(matrix);

  return std::make_tuple(matrix,header);
}
//-----------------------------------------------------------------------
// bulk_load : Can be usefull for functional analysis 
//  * Only for regression data
//-----------------------------------------------------------------------
arma::mat BulkLoad ( const std::filesystem::path& path,
                     const bool& transpose )
{
  arma::mat data, temp;
  size_t counter = 0;
  for (const auto & entry : std::filesystem::recursive_directory_iterator(path))
  {
    if ( std::filesystem::is_regular_file(entry) )
    {
      temp = Load(entry.path(), transpose);
      if ( counter == 0 )
        data = temp;
      else
      {
        if ( transpose )
          data = arma::join_cols(data,temp.rows(1,temp.n_rows-1));
        else 
          data = arma::join_rows(data,temp.cols(1,temp.n_cols-1));
      }
      counter++;
    }
  }
  return data;
}

//-----------------------------------------------------------------------
// BulkLoadSplit: Can be usefull for functional analysis 
//  * Only for regression data
//-----------------------------------------------------------------------
std::tuple<arma::mat,arma::mat> 
          BulkLoadSplit ( const std::filesystem::path& path,
                          const double& test_size,
                          const bool& transpose )
{
  arma::mat data, train_data, test_data, temp, inp, labs;
  size_t num_files = 0;
  for (const auto & entry : std::filesystem::recursive_directory_iterator(path))
  {
    if ( std::filesystem::is_regular_file(entry) )
    {
      num_files++;
    }
  }

  size_t counter = 0;
  for (const auto & entry : std::filesystem::recursive_directory_iterator(path))
  {
    if ( std::filesystem::is_regular_file(entry) )
    {
      temp = Load(entry.path(), transpose);
      if ( counter == 0 )
        data = temp;
      else
      {
        if ( transpose )
          data = arma::join_cols(data,temp.rows(1,temp.n_rows-1));
        else 
          data = arma::join_rows(data,temp.cols(1,temp.n_cols-1));
      }
      counter++;
    }
  }

  if ( transpose )
  {
    inp = data.row(0);
    data = data.rows(1,data.n_rows-1);
    arma::inplace_trans(data);
    mlpack::data::Split(data,train_data,test_data,test_size);
    arma::inplace_trans(train_data);
    arma::inplace_trans(test_data);
    train_data = arma::join_cols(inp, train_data);
    test_data = arma::join_cols(inp, test_data);
  }
  else
  {
    inp = data.col(0);
    data = data.cols(1,data.n_cols-1);
    mlpack::data::Split(data,train_data,test_data,test_size);
    train_data = arma::join_rows(inp, train_data);
    test_data = arma::join_rows(inp, test_data);
  }
  return std::make_tuple(train_data, test_data);
}

//-----------------------------------------------------------------------
// BulkLoadSplit2: Can be usefull for functional analysis this version is 
//  for header included files.
//  * Only for regression data
//-----------------------------------------------------------------------
std::tuple<arma::mat,arma::mat> 
          BulkLoadSplit2 ( const std::filesystem::path& path,
                          const double& test_size,
                          const bool& transpose )
{
  arma::mat data, train_data, test_data, temp, inp, labs;
  size_t num_files = 0;
  for (const auto & entry : std::filesystem::recursive_directory_iterator(path))
  {
    if ( std::filesystem::is_regular_file(entry) )
    {
      num_files++;
    }
  }

  size_t counter = 0;
  for (const auto & entry : std::filesystem::recursive_directory_iterator(path))
  {
    if ( std::filesystem::is_regular_file(entry) )
    {
      temp = Load(entry.path(), transpose);
      if ( counter == 0 )
        data = temp;
      else
      {
        if ( transpose )
          data = arma::join_cols(data,temp.rows(1,temp.n_rows-1));
        else 
          data = arma::join_rows(data,temp.cols(1,temp.n_cols-1));
      }
      counter++;
    }
  }

  if ( transpose )
  {
    inp = data.row(0);
    data = data.rows(1,data.n_rows-1);
    arma::inplace_trans(data);
    mlpack::data::Split(data,train_data,test_data,test_size);
    arma::inplace_trans(train_data);
    arma::inplace_trans(test_data);
    train_data = arma::join_cols(inp, train_data);
    test_data = arma::join_cols(inp, test_data);
  }
  else
  {
    inp = data.col(0);
    data = data.cols(1,data.n_cols-1);
    mlpack::data::Split(data,train_data,test_data,test_size);
    train_data = arma::join_rows(inp, train_data);
    test_data = arma::join_rows(inp, test_data);
  }
  return std::make_tuple(train_data, test_data);
}

//-----------------------------------------------------------------------------
// loadCSV  : Legacy loader for previous versions
//-----------------------------------------------------------------------------
arma::mat loadCSV ( const std::string &filename, 
                    const std::string &delimeter = "," )
{
  std::ifstream csv(filename);
  std::vector<std::vector<double>> datas;

  for ( std::string line; std::getline(csv, line); )
  {
      std::vector<double> data;

      // split string by delimeter
      auto start = 0;
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

//-----------------------------------------------------------------------------
// Load : Legacy Load
//-----------------------------------------------------------------------------
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
