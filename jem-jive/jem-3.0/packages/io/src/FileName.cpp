
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jem, a general purpose programming toolkit.
 *
 *  Commercial License Usage
 *
 *  This file may be used under the terms of a commercial license
 *  provided with the software, or under the terms contained in a written
 *  agreement between you and DRG. For more information contact DRG at
 *  http://www.dynaflow.com.
 *
 *  GNU Lesser General Public License Usage
 *
 *  Alternatively, this file may be used under the terms of the GNU
 *  Lesser General Public License version 2.1 or version 3 as published
 *  by the Free Software Foundation and appearing in the file
 *  LICENSE.LGPLv21 and LICENSE.LGPLv3 included in the packaging of this
 *  file. Please review the following information to ensure the GNU
 *  Lesser General Public License requirements will be met:
 *  https://www.gnu.org/licenses/lgpl.html and
 *  http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html.
 *
 *  Jem version: 3.0
 *  Date:        Fri 20 Dec 14:27:58 CET 2019
 */


#include <cctype>
#include <jem/base/System.h>
#include <jem/base/StringBuffer.h>
#include <jem/util/Flex.h>
#include <jem/io/IOException.h>
#include <jem/io/File.h>
#include <jem/io/FileName.h>


JEM_BEGIN_PACKAGE( io )


using jem::util::Flex;


//=======================================================================
//   class FileName
//=======================================================================

//-----------------------------------------------------------------------
//   static data members
//-----------------------------------------------------------------------


#if defined(JEM_FS_UNIX)

const char FileName::PATH_SEP_   = ':';
const char FileName::FILE_SEP_   = '/';
const char FileName::SUFFIX_SEP_ = '.';

#elif defined(JEM_FS_WINDOWS)

const char FileName::PATH_SEP_   = ';';
const char FileName::FILE_SEP_   = '\\';
const char FileName::SUFFIX_SEP_ = '.';

#else
#  error "Unsupported file system type -- check jem/defines.h"
#endif


//-----------------------------------------------------------------------
//   (disabled) constructors & destructor
//-----------------------------------------------------------------------


FileName::FileName ()
{}


FileName::~FileName ()
{}


//-----------------------------------------------------------------------
//   pathSeparator, fileSeparator & suffixSeparator
//-----------------------------------------------------------------------


char FileName::pathSeparator ()
{
  return PATH_SEP_;
}


char FileName::fileSeparator ()
{
  return FILE_SEP_;
}


char FileName::suffixSeparator ()
{
  return SUFFIX_SEP_;
}


//-----------------------------------------------------------------------
//   isAbsolute
//-----------------------------------------------------------------------


bool FileName::isAbsolute ( const String& name )
{
  return ( skipRoot( name ) > 0 );
}


//-----------------------------------------------------------------------
//   isRoot
//-----------------------------------------------------------------------


bool FileName::isRoot ( const String& name )
{
  idx_t  n = skipRoot ( name );

  return ( n > 0 && n == name.size() );
}


//-----------------------------------------------------------------------
//   getFileName
//-----------------------------------------------------------------------


String FileName::getFileName ( const String& name )
{
  idx_t  i = name.rfind ( FILE_SEP_ );
  idx_t  j = skipRoot   ( name );

  if ( i < j )
  {
    return name[slice(j,END)];
  }
  else
  {
    return name[slice(i + 1,END)];
  }
}


//-----------------------------------------------------------------------
//   getDirName
//-----------------------------------------------------------------------


String FileName::getDirName ( const String& name )
{
  idx_t  i = name.rfind ( FILE_SEP_ );
  idx_t  j = skipRoot   ( name );

  return name[slice(BEGIN,max(i,j))];
}


//-----------------------------------------------------------------------
//   getSuffix
//-----------------------------------------------------------------------


String FileName::getSuffix ( const String& name )
{
  idx_t  i = findSuffix ( name );

  if ( i < 0 )
  {
    return String ();
  }
  else
  {
    return name[slice(i,END)];
  }
}


//-----------------------------------------------------------------------
//   getBaseFileName
//-----------------------------------------------------------------------


String FileName::getBaseFileName ( const String& name )
{
  idx_t  i = name.rfind ( FILE_SEP_ );
  idx_t  j = skipRoot   ( name );
  idx_t  k = name.rfind ( SUFFIX_SEP_ );

  if ( i < j )
  {
    i = j;
  }
  else
  {
    i = i + 1;
  }
  if ( k < i )
  {
    k = name.size();
  }

  return name[slice(i,k)];
}


//-----------------------------------------------------------------------
//   getAbsoluteName
//-----------------------------------------------------------------------


String FileName::getAbsoluteName ( const String& name )
{
  if ( name.size() == 0 || isAbsolute( name ) )
  {
    return name;
  }
  else
  {
    return simplify ( join( File::getCwd(), name ) );
  }
}


//-----------------------------------------------------------------------
//   getCanonicalName
//-----------------------------------------------------------------------


String FileName::getCanonicalName ( const String& name )
{
  if ( name.size() == 0 )
  {
    return String ();
  }

  String  fname ( getFileName ( name ) );
  String  dname ( getDirName  ( name ) );


  if ( dname.size() == 0 )
  {
    dname = File::getCwd ();
  }
  else
  {
    String  cwd = File::getCwd ();

    try
    {
      File::setCwd ( dname );
    }
    catch ( const IOException& )
    {
      return String ();
    }

    dname = File::getCwd ();

    File::setCwd ( cwd );
  }

  return join ( dname, fname );
}


//-----------------------------------------------------------------------
//   getParent
//-----------------------------------------------------------------------


String FileName::getParent ( const String& name )
{
  if ( isRoot( name ) )
  {
    return String ();
  }
  else
  {
    return getDirName ( name );
  }
}


//-----------------------------------------------------------------------
//   translate
//-----------------------------------------------------------------------


String FileName::translate ( const String& in )
{
  String  s = in.stripWhite ();

#if defined(JEM_FS_UNIX)

  // On Unix, replace a leading "~/" by the user's home directory.

  if ( s.size() > 1 && s[0] == '~' && s[1] == '/' )
  {
    StringBuffer  buf ( System::getProperty( "user.home" ) );

    if ( buf.size() > 0 && buf.back() != '/' )
    {
      buf += s[slice(1,END)];
    }
    else
    {
      buf += s[slice(2,END)];
    }

    s = buf.toString ();
  }

#else

  // Translate Unix style separators to native separator strings.

  if ( s.find( '/' ) >= 0 )
  {
    StringBuffer  buf ( s );

    const idx_t   len = buf.size ();
    char*         ptr = buf.addr ();

    for ( idx_t i = 0; i < len; i++ )
    {
      if ( ptr[i] == '/' )
      {
        ptr[i] = FILE_SEP_;
      }
    }

    s = buf.toString ();
  }

#endif

  return s;
}


//-----------------------------------------------------------------------
//   simplify
//-----------------------------------------------------------------------


String FileName::simplify ( const String& s )
{
  String        name = s.stripWhite ();

  StringBuffer  buf;
  Flex<idx_t>   index;

  bool          isAbsPath;
  int           level;
  idx_t         i, j, k;


  // Handle the trivial case.

  if ( name.size() == 0 )
  {
    return String ();
  }

  //
  // The array `index' stores the pointers to the directories/files
  // that should be included in the simplified name.
  // If i = index[k] and  j = index[k+1] (k is even), then name[i]
  // is the first character of the k-th directory, and name[j]
  // is one position beyond the last character in the k-th directory.
  // Separator strings are not included, except when they are part
  // of the root directory. Example:
  //
  //   name  = "/usr/lib/../local/./lib/a.out"
  //   index = [ 0, 1, 1, 4, 12, 17, 20, 23, 24, 29 ]
  //

  buf  .reserve ( name.size() );
  index.reserve ( name.size() );

  // First, find the root directory, if it is present.

  i = 0;
  j = skipRoot ( name );

  if ( j > 0 )
  {
    // OK, the name does contain the root directory. Push it onto
    // the index stack.

    index.pushBack ( 0 );
    index.pushBack ( j );

    isAbsPath = true;
  }
  else
  {
    isAbsPath = false;
  }

  // Scan the remainder of the name and build the index array.

  level = 0;
  i     = j;

  while ( i < name.size() )
  {
    j = name.find ( FILE_SEP_, i );

    if ( j < 0 )
    {
      j = name.size( );
    }

    k = j - i;

    // What type of directory is this?

    if      ( k == 0 )
    {
      ; // This is an empty directory, so ignore it
    }
    else if ( k == 1 && name[i] == '.' )
    {
      ; // This is the current directory ".", so ignore it
    }
    else if ( k == 2 && name[i] == '.' && name[i+1] == '.' )
    {
      // This is one directory up ".."

      if ( level <= 0 )
      {
        // Make sure we do not go up beyond the root directory.

        if ( ! isAbsPath )
        {
          index.pushBack ( i );
          index.pushBack ( j );
          level--;
        }
      }
      else
      {
        // Pop the last two indices from the stack.

        index.popBack ();
        index.popBack ();
        level--;
      }
    }
    else
    {
      // This is a regular directory. Push it onto the stack, and
      // descend one level.

      index.pushBack ( i );
      index.pushBack ( j );
      level++;
    }

    // Proceed with the next directory

    i = j + 1;
  }

  // Concatenate all directories to obtain the simplified name.

  for ( i = 0; i < index.size(); i += 2 )
  {
    for ( j = index[i]; j < index[i + 1]; j++ )
    {
      buf += name[j];
    }

    // Add a separator string, unless: (i) we have just written
    // the root directory into the buffer, and (ii) this is the last
    // directory that will be written into the buffer

    if ( i < (index.size() - 2) && !(i == 0 && isAbsPath) )
    {
      buf += FILE_SEP_;
    }
  }

  // If we get here and the name is empty, it must be the current
  // directory.

  if ( buf.size() == 0 )
  {
    buf += '.';
  }

  return buf.toString ();
}


//-----------------------------------------------------------------------
//   split
//-----------------------------------------------------------------------


Array<String> FileName::split ( const String& name )
{
  Flex<String>  parts;
  idx_t         i, j;


  parts.reserve ( 8 );

  j = skipRoot ( name );

  if ( j > 0 )
  {
    parts.pushBack ( name[slice(BEGIN,j)] );
  }

  i = j;

  while ( i < name.size() )
  {
    j = name.find ( FILE_SEP_, i );

    if ( j < 0 )
    {
      j = name.size();
    }
    if ( j > i )
    {
      parts.pushBack ( name[slice(i,j)] );
    }

    i = j + 1;
  }

  return Array<String> ( parts.begin(), parts.end() );
}


//-----------------------------------------------------------------------
//   join
//-----------------------------------------------------------------------


String FileName::join ( const String& part1, const String& part2 )
{
  if ( part1.size() == 0 )
  {
    return part2;
  }
  if ( part2.size() == 0 )
  {
    return part1;
  }
  if ( isRoot( part1 ) )
  {
    return (part1 + part2);
  }
  else
  {
    return (part1 + String( FILE_SEP_ ) + part2);
  }
}


String FileName::join ( const Array<String>& parts )
{
  StringBuffer  buf;
  Flex<idx_t>   index;

  bool          isAbsPath;
  idx_t         i, n;


  index.reserve ( parts.size() );

  // Compile a list of all non-empty parts.

  for ( i = 0; i < parts.size(); i++ )
  {
    if ( parts[i].size() > 0 )
    {
      index.pushBack ( i );
    }
  }

  n = index.size ();

  if ( n == 0 )
  {
    return String ();
  }

  buf.reserve ( 128 );

  isAbsPath = isRoot ( parts[index[0]] );

  for ( i = 0; i < n; i++ )
  {
    buf += parts[index[i]];

    if ( i < n - 1 && ! (i == 0 && isAbsPath) )
    {
      buf += FILE_SEP_;
    }
  }

  return buf.toString ();
}


//-----------------------------------------------------------------------
//   skipRoot
//-----------------------------------------------------------------------


idx_t FileName::skipRoot ( const String& name )
{

#if defined(JEM_FS_UNIX)

  if ( name.size() > 0 && name[0] == '/' )
  {
    return 1_idx;
  }
  else
  {
    return 0_idx;
  }

#elif defined(JEM_FS_WINDOWS)

  if ( name.size() > 0 && name[0] == '\\' )
  {
    return 1_idx;
  }
  else if ( name.size() > 2 && std::isalpha(name[0]) &&
            name[1] == ':'  && name[2] == '\\' )
  {
    return 3_idx;
  }
  else
  {
    return 0_idx;
  }

#else
#  error "Unsupported file system -- check defines.h"
#endif

}


//-----------------------------------------------------------------------
//   findSuffix
//-----------------------------------------------------------------------


idx_t FileName::findSuffix ( const String& name )
{
  idx_t  i = name.rfind ( FILE_SEP_ );
  idx_t  j = skipRoot   ( name );

  if ( i < j )
  {
    i = j;
  }

  j = name.rfind ( SUFFIX_SEP_ );

  if ( j < i )
  {
    return -1;
  }
  else
  {
    return j;
  }
}


JEM_END_PACKAGE( io )
