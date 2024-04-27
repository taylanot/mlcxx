/**
 * @file system.h
 * @author Ozgur Taylan Turan
 *
 * Handling system level stuff.
 *
 * TODO: 
 *
 *
 */

#ifndef DATA_MANI_H
#define DATA_MANI_H

namespace utils {

//=============================================================================
//  split_path 
//=============================================================================
std::vector<std::filesystem::path> split_path 
                                ( const std::filesystem::path& paths )
{
  std::vector<std::filesystem::path> list;
  for ( auto& path: paths )
  {
    list.push_back(path); 
  }
  return list;
}

//=============================================================================
//  split_path 
//=============================================================================
std::vector<std::string> split_path 
                                ( const std::string& paths )
{
  std::filesystem::path temp = paths;
  std::vector<std::string> list;
  for ( auto& path: temp)
  {
    list.push_back(path.string()); 
  }
  return list;
}

//=============================================================================
// remove_path 
//=============================================================================
std::filesystem::path remove_path ( const std::filesystem::path& path,
                                    const std::filesystem::path& path_rem )
{
  auto dirs = split_path(path);
  auto remdirs = split_path(path_rem);

  for ( auto rem: remdirs )
  {
   auto it = std::find(dirs.begin(), dirs.end(), rem);
   dirs.erase(it);
  }

  std::filesystem::path update;
  for ( auto dir: dirs )
  {
   update /= dir;
  }
  return update;
}

////=============================================================================
//// to_string
////=============================================================================
//std::string to_string ( const jem::String& data )
//{
//  const char* dataptr = data.addr();
//  return std::string(dataptr, data.size());
//}
//
////=============================================================================
//// to_char
////=============================================================================
//const char* to_char ( const jem::String& data )
//{
//  const char* dataptr = data.addr();
//  return dataptr;
//}
} // namespace utils

#endif 
