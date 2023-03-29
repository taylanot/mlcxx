
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


#include <cstring>
#include <jem/base/limits.h>
#include <jem/base/StringBuffer.h>
#include <jem/base/native/System.h>
#include <jem/base/native/Time.h>
#include <jem/io/FileName.h>
#include <jem/io/FileFlags.h>
#include <jem/io/IOException.h>
#include <jem/io/FileInputStream.h>
#include <jem/io/FileOutputStream.h>
#include "native/WinFile.h"
#include "native/WinFileStream.h"

extern "C"
{
  #include <windows.h>
}

#include <jem/base/native/winclean.h>


JEM_BEGIN_PACKAGE( io )


//=======================================================================
//   class WinFile
//=======================================================================

//-----------------------------------------------------------------------
//   erase
//-----------------------------------------------------------------------


void WinFile::erase ( const String& name )
{
  char    path[MAX_PATH + 1];
  HANDLE  hfile;

  copyName ( path, name );

  // Get a handle for the file.

  hfile = CreateFileA ( path,
                        (DWORD) 0x0,
                        (DWORD) 0x0,
                        nullptr,
                        OPEN_EXISTING,
                        FILE_ATTRIBUTE_READONLY |
                        FILE_FLAG_BACKUP_SEMANTICS,
                        nullptr );

  if ( hfile == INVALID_HANDLE_VALUE )
  {
    goto error;
  }
  else
  {
    BY_HANDLE_FILE_INFORMATION  st;
    BOOL                        result;

    result = GetFileInformationByHandle ( hfile, &st );

    CloseHandle ( hfile );

    if ( ! result )
    {
      goto error;
    }

    if ( st.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY )
    {
      result = RemoveDirectoryA ( path );
    }
    else
    {
      result = DeleteFileA      ( path );
    }

    if ( ! result )
    {
      goto error;
    }
  }

  return;

 error:

  throw IOException (
    JEM_FUNC,
    String::format (
      "error deleting `%s\' (%s)",
      name,
      NativeSystem::strerror ()
    )
  );
}


//-----------------------------------------------------------------------
//   getInfo
//-----------------------------------------------------------------------


void WinFile::getInfo

  ( FileInfo&      info,
    const String&  name )

{
  char    path[MAX_PATH + 1];
  HANDLE  hfile;

  copyName ( path, name );

  // Get a handle for the file.

  hfile = CreateFileA ( path,
                        (DWORD) 0x0,
                        (DWORD) 0x0,
                        nullptr,
                        OPEN_EXISTING,
                        FILE_ATTRIBUTE_READONLY |
                        FILE_FLAG_BACKUP_SEMANTICS,
                        nullptr );

  if ( hfile != INVALID_HANDLE_VALUE )
  {
    BY_HANDLE_FILE_INFORMATION  st;
    LARGE_INTEGER               size;

    if ( GetFileInformationByHandle( hfile, &st ) )
    {
      if ( st.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY )
      {
        info.mode |= File::IS_DIR;
      }
      else
      {
        info.mode |= File::IS_FILE;
      }

      if ( st.dwFileAttributes & FILE_ATTRIBUTE_READONLY )
      {
        info.mode |= File::IS_READABLE;
      }
      else
      {
        info.mode |= File::IS_READABLE | File::IS_WRITABLE;
      }

      info.atime = WinTime::getTime ( st.ftLastAccessTime );
      info.mtime = WinTime::getTime ( st.ftLastWriteTime );
    }

    if ( GetFileSizeEx( hfile, &size ) )
    {
      if ( size.QuadPart > maxOf( info.length ) )
      {
        info.length = -1;
      }
      else
      {
        info.length = (lint) size.QuadPart;
      }
    }

    CloseHandle ( hfile );
  }
}


//-----------------------------------------------------------------------
//   makeDir
//-----------------------------------------------------------------------


void WinFile::makeDir ( const String& name )
{
  char  path[MAX_PATH + 1];

  copyName ( path, name );

  if ( ! CreateDirectoryA( path, nullptr ) )
  {
    String  details =  NativeSystem::strerror ();

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


Ref<FileStream> WinFile::makeTemp

  ( String&    name,
    OpenFlags  flags )

{
  char  tmpName[MAX_PATH + 1];
  char  tmpPath[MAX_PATH + 1];
  char  prefix [MAX_PATH + 1];


  copyName ( tmpPath, FileName::getDirName     ( name ) );
  copyName ( prefix,  FileName::getBaseFileName( name ) );

  if ( ! GetTempFileNameA( tmpPath, prefix, 0, tmpName ) )
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

  name   = clone ( tmpName );
  flags &= FileFlags::WRITE;

  return newInstance<WinFileStream> ( name, flags );
}


//-----------------------------------------------------------------------
//   getCwd
//-----------------------------------------------------------------------


String WinFile::getCwd ()
{
  char   path[MAX_PATH + 1];

  DWORD  len = GetCurrentDirectoryA ( MAX_PATH, path );

  if ( len == 0 )
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

  if ( len > MAX_PATH )
  {
    SetLastError ( ERROR_BUFFER_OVERFLOW );

    throw IOException (
      JEM_FUNC,
      "failed to get the working directory name; name too long"
    );
  }

  return clone ( path );
}


//-----------------------------------------------------------------------
//   setCwd
//-----------------------------------------------------------------------


void WinFile::setCwd ( const String& name )
{
  char  path[MAX_PATH + 1];

  copyName ( path, name );

  if ( ! SetCurrentDirectoryA( path ) )
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
//   copyName
//-----------------------------------------------------------------------


int WinFile::copyName

  ( char*          path,
    const String&  name )

{
  idx_t  len = name.size ();

  if ( len > MAX_PATH )
  {
    SetLastError ( ERROR_BUFFER_OVERFLOW );

    throw IOException (
      JEM_FUNC,
      String::format (
        "file name (%s) too long", name
      )
    );
  }

  std::memcpy ( path, name.addr(), len );

  path[len] = '\0';

  return len;
}


//-----------------------------------------------------------------------
//   listRoots
//-----------------------------------------------------------------------


Array<String> WinFile::listRoots ()
{
  StringBuffer  buf ( MAX_PATH + 1 );
  int           len;


  len = (int) GetLogicalDriveStringsA ( MAX_PATH, buf.addr() );

  if ( len > MAX_PATH )
  {
    buf.resize ( len + 1 );

    len = (int) GetLogicalDriveStringsA ( (DWORD) len, buf.addr() );
  }

  if ( len > buf.size() )
  {
    SetLastError ( ERROR_BUFFER_OVERFLOW );

    throw IOException (
      JEM_FUNC,
      String::format (
        "failed to list the root directories; "
        "not enough space to store the logical drive names"
      )
    );
  }

  // Count the number of logical drives.

  int   i, j, k;

  for ( j = k = 0; j < len; j++ )
  {
    if ( buf[j] == '\0' )
    {
      k++;
    }
  }

  Array<String>  list ( k );

  i = k = 0;

  for ( j = 0; j < len; j++ )
  {
    if ( buf[j] == '\0' )
    {
      list[k++] = buf[slice(i,j)];
      i         = j + 1;
    }
  }

  return list;
}


//-----------------------------------------------------------------------
//   getStdin
//-----------------------------------------------------------------------


Ref<InputStream> WinFile::getStdin ()
{
  return newInstance<FileInputStream> (
    newInstance<WinFileStream> (
      (void*) GetStdHandle ( STD_INPUT_HANDLE ),
      FileFlags::READ
    )
  );
}


//-----------------------------------------------------------------------
//   getStdout
//-----------------------------------------------------------------------


Ref<OutputStream> WinFile::getStdout ()
{
  return newInstance<FileOutputStream> (
    newInstance<WinFileStream> (
      (void*) GetStdHandle ( STD_OUTPUT_HANDLE ),
      FileFlags::WRITE
    )
  );
}


//-----------------------------------------------------------------------
//   getStderr
//-----------------------------------------------------------------------


Ref<OutputStream> WinFile::getStderr ()
{
  return newInstance<FileOutputStream> (
    newInstance<WinFileStream> (
      (void*) GetStdHandle ( STD_ERROR_HANDLE ),
      FileFlags::WRITE
    )
  );
}


JEM_END_PACKAGE( io )
