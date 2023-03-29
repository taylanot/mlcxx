
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


#include <cstdlib>
#include <jem/base/limits.h>
#include <jem/base/CString.h>
#include <jem/base/StringBuffer.h>
#include <jem/base/native/System.h>
#include <jem/io/FileFlags.h>
#include <jem/io/IOException.h>
#include <jem/io/FileInputStream.h>
#include <jem/io/FileOutputStream.h>
#include "native/PosixFile.h"
#include "native/PosixFileStream.h"

extern "C"
{
  #include <stdio.h>
  #include <sys/stat.h>
  #include <sys/types.h>
  #include <unistd.h>
  #include <fcntl.h>
}


#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wstrict-overflow"
#endif


JEM_BEGIN_PACKAGE( io )


//=======================================================================
//   class PosixFile
//=======================================================================

//-----------------------------------------------------------------------
//   erase
//-----------------------------------------------------------------------


void PosixFile::erase ( const String& name )
{
  if ( ::remove( makeCString( name ) ) )
  {
    String  details = NativeSystem::strerror ();

    throw IOException (
      JEM_FUNC,
      String::format (
        "error deleting `%s\' (%s)",
        name,
        details
      )
    );
  }
}


//-----------------------------------------------------------------------
//   getInfo
//-----------------------------------------------------------------------


void PosixFile::getInfo

  ( FileInfo&      info,
    const String&  name )

{
  struct stat  st;

  if ( ::stat( makeCString( name ), &st ) == 0 )
  {
    if ( S_ISREG( st.st_mode ) )
    {
      info.mode |= File::IS_FILE;
    }
    else if ( S_ISDIR( st.st_mode ) )
    {
      info.mode |= File::IS_DIR;
    }
    else if ( S_ISLNK( st.st_mode ) )
    {
      info.mode |= File::IS_LINK;
    }

    if ( st.st_mode & S_IRUSR )
    {
      info.mode |= File::IS_READABLE;
    }
    if ( st.st_mode & S_IWUSR )
    {
      info.mode |= File::IS_WRITABLE;
    }
    if ( st.st_mode & S_IXUSR )
    {
      info.mode |= File::IS_EXECUTABLE;
    }

    if ( info.mode & File::IS_FILE )
    {
      if ( st.st_size > maxOf( info.length ) )
      {
        info.length = -1;
      }
      else
      {
        info.length = (lint) st.st_size;
      }
    }

    info.atime = Time( (lint) st.st_atime );
    info.mtime = Time( (lint) st.st_mtime );
  }
}


//-----------------------------------------------------------------------
//   makeDir
//-----------------------------------------------------------------------


void PosixFile::makeDir ( const String& name )
{
  if ( ::mkdir( makeCString( name ), 0755 ) )
  {
    String  details = NativeSystem::strerror ();

    throw IOException (
      JEM_FUNC,
      String::format (
        "error creating directory `%s\' (%s)",
        name,
        details
      )
    );
  }
}


//-----------------------------------------------------------------------
//   makeTemp
//-----------------------------------------------------------------------


Ref<FileStream> PosixFile::makeTemp

  ( String&    name,
    OpenFlags  flags )

{
  StringBuffer  buf ( name );
  idx_t         len;
  int           fd;


  buf += "XXXXXX";
  len  = buf.size ();
  buf += '\0';
  fd   = ::mkstemp ( buf.addr() );

  if ( fd == -1 )
  {
    String  details = NativeSystem::strerror ();

    throw IOException (
      JEM_FUNC,
      String::format (
        "error creating temporary file `%s\' (%s)",
        name,
        details
      )
    );
  }

  name   = buf[slice(BEGIN,len)];
  flags &= FileFlags::WRITE;

  return newInstance<PosixFileStream> ( fd, flags );
}


//-----------------------------------------------------------------------
//   getCwd
//-----------------------------------------------------------------------


String PosixFile::getCwd ()
{
  char  buf[256 + 1];

  if ( ::getcwd( buf, (int) (sizeof(buf) - 1) ) == nullptr )
  {
    String  details = NativeSystem::strerror ();

    throw IOException (
      JEM_FUNC,
      String::format (
        "failed to get the working directory name (%s)",
        details
      )
    );
  }

  return clone ( buf );
}


//-----------------------------------------------------------------------
//   setCwd
//-----------------------------------------------------------------------


void PosixFile::setCwd ( const String& path )
{
  if ( ::chdir( makeCString( path ) ) )
  {
    String  details = NativeSystem::strerror ();

    throw IOException (
      JEM_FUNC,
      String::format (
        "failed to set the working directory (%s)",
        details
      )
    );
  }
}


//-----------------------------------------------------------------------
//   listRoots
//-----------------------------------------------------------------------


Array<String> PosixFile::listRoots ()
{
  Array<String>  list ( 1 );

  list[0] = "/";

  return list;
}


//-----------------------------------------------------------------------
//   getStdin
//-----------------------------------------------------------------------


Ref<InputStream> PosixFile::getStdin ()
{
  return newInstance<FileInputStream> (
    newInstance<PosixFileStream> ( 0, FileFlags::READ )
  );
}


//-----------------------------------------------------------------------
//   getStdout
//-----------------------------------------------------------------------


Ref<OutputStream> PosixFile::getStdout ()
{
  return newInstance<FileOutputStream> (
    newInstance<PosixFileStream> ( 1, FileFlags::WRITE )
  );
}


//-----------------------------------------------------------------------
//   getStderr
//-----------------------------------------------------------------------


Ref<OutputStream> PosixFile::getStderr ()
{
  return newInstance<FileOutputStream> (
    newInstance<PosixFileStream> ( 2, FileFlags::WRITE )
  );
}


JEM_END_PACKAGE( io )
