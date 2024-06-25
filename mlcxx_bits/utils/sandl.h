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
template<class T, class O=DTYPE>
arma::Mat<O> Combine ( const T& data1,
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
template<class T, class O=DTYPE>
arma::Mat<O> Load ( const T& filename,
                    const bool& transpose,
                    const bool& count = false )
{
  arma::Mat<O> matrix;
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
template<class T,class O=DTYPE>
std::tuple<arma::Mat<O>,arma::field<std::string>> 
                                              LoadwHeader ( const T& filename )
{
  arma::Mat<O> matrix;
  arma::field<std::string> header;
  matrix.load(arma::csv_name(filename, header));
  return std::make_tuple(matrix,header);
}
//-----------------------------------------------------------------------
// bulk_load : Can be usefull for functional analysis 
//  * Only for regression data
//-----------------------------------------------------------------------
template<class T,class O=DTYPE>
arma::Mat<O> BulkLoad ( const std::filesystem::path& path,
                        const bool& transpose )
{
  arma::Mat<O> data, temp;
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
template<class O=DTYPE>
std::tuple<arma::Mat<O>,arma::Mat<O>> 
          BulkLoadSplit ( const std::filesystem::path& path,
                          const double& test_size,
                          const bool& transpose )
{
  arma::Mat<O> data, train_data, test_data, temp, inp, labs;
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
// BulkLoadSplit: Can be usefull for functional analysis this version is 
//  for header included files.
//  * Only for regression data
//-----------------------------------------------------------------------
template<class O=DTYPE>
std::tuple<arma::Mat<O>,arma::Mat<O>,
           arma::field<std::string>,arma::field<std::string>>
          BulkLoadSplit ( const std::filesystem::path& path,
                          const double& test_size )
{
  arma::Mat<O> data, train_data, test_data, temp, inp, labs;
  arma::field<std::string> temp_header;
  std::string first_header;
  size_t train_size_ = 0; 
  size_t test_size_ = 0; 
  size_t num_files = 0;
  for (const auto & entry : std::filesystem::recursive_directory_iterator(path))
  {
    if ( std::filesystem::is_regular_file(entry) )
    {
      num_files++;
    }
  }

  arma::field<arma::field<std::string>> train_headers(num_files);
  arma::field<arma::field<std::string>> test_headers(num_files);
  size_t counter = 0;
  for (const auto & entry : std::filesystem::recursive_directory_iterator(path))
  {
    if ( std::filesystem::is_regular_file(entry) )
    {
      auto temp_load = LoadwHeader(entry.path());
      temp = std::get<0>(temp_load);
      arma::Mat<O> x = temp.col(0);
      arma::Mat<O> y = temp.cols(1,temp.n_cols-1);
      temp_header = std::get<1>(temp_load);
      first_header = temp_header(0);
      temp_header = temp_header.cols(1,temp_header.n_cols-1);


      size_t num_train = (1-test_size)*y.n_cols;

      arma::uvec idx = arma::randperm(y.n_cols);

      size_t num_test = y.n_cols - num_train;

      if (counter == 0)
      {
        train_data = y.cols(idx(arma::span(0,num_train-1)));
        test_data = y.cols(idx(arma::span(num_train,idx.n_elem-1)));
      }

      else
      {
        train_data = arma::join_rows(train_data,
            y.cols(idx(arma::span(0,num_train-1))));
        test_data = arma::join_rows(test_data,
            y.cols(idx(arma::span(num_train,idx.n_elem-1))));
      }

      arma::field<std::string> temp_train(1,num_train);
      arma::field<std::string> temp_test(1,num_test);
      
      size_t c1 = 0;
      size_t c2 = 0;

      for (size_t i=0; i<idx.n_rows;i++)
      {
        if (i<num_train)
        {
          temp_train(c1++) = temp_header(idx(i));
        }
        else
        {
          temp_test(c2++) = temp_header(idx(i));
        }
      }

      train_size_ += num_train;
      test_size_ += num_test;
      train_headers(counter) = temp_train;
      test_headers(counter) = temp_test;
      counter++;
    }

  }

  arma::field<std::string> train_header(1,train_size_+1);
  arma::field<std::string> test_header(1,test_size_+1);
  train_header(0)=first_header;test_header(0)=first_header;
  size_t c1 = 1;
  size_t c2 = 1;

  for (size_t i=0; i<num_files;i++)
  {
    for (size_t j=0; j<train_headers(i).n_cols;j++)
    {
      train_header(c1++) = train_headers(i).at(j);
    }


    for (size_t j=0; j<test_headers(i).n_cols;j++)
    {
      test_header(c2++) = test_headers(i).at(j);
    }
  }

  inp = temp.col(0);
  train_data = arma::join_rows(inp, train_data);
  test_data = arma::join_rows(inp, test_data);

  return std::make_tuple(train_data,test_data,train_header,test_header);
}

//-----------------------------------------------------------------------------
// loadCSV  : Legacy loader for previous versions
//-----------------------------------------------------------------------------
template<class O=DTYPE>
arma::Mat<O> loadCSV ( const std::string &filename, 
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

  arma::Mat<O> data_mat = arma::zeros<arma::Mat<O>>(datas.size(), datas[0].size());

  for (size_t i=0; i<datas.size(); i++) {
      arma::Mat<O> r(datas[i]);
      data_mat.row(i) = r.t();
  }

  return data_mat;
}

} // namespace utils

#endif
