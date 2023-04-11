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

#ifndef SYSTEM_H
#define SYSTEM_H

namespace utils {

//-----------------------------------------------------------------------
//  create_dirs
//-----------------------------------------------------------------------

void create_dirs( const std::string& dirs )
{
  std::string command = "mkdir -p "+dirs;
  const int dir_err = system(command.c_str());
  std::string err_msg = "Something went wrong with mkdir!";
  if ( dir_err != 0)
   BOOST_THROW_EXCEPTION( std::runtime_error( err_msg ) );
}

} // namespace utils

#endif 
