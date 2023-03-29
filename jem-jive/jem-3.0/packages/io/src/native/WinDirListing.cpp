
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
#include <jem/base/native/System.h>
#include <jem/io/IOException.h>
#include "native/WinDirListing.h"

extern "C"
{
  #include <windows.h>
}

#include <jem/base/native/winclean.h>


JEM_BEGIN_PACKAGE( io )


//=======================================================================
//   class WinDirListing
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


WinDirListing::WinDirListing ( const String& name ) :

  name_ ( name )

{
  find_ = (void*) INVALID_HANDLE_VALUE;

  if ( name.size() > (MAX_PATH - 2) )
  {
    throw IOException (
      JEM_FUNC,
      String::format (
        "directory name `%s\' too long; max length is %d",
        name,
        (int) (MAX_PATH - 2)
      )
    );
  }
}


WinDirListing::~WinDirListing ()
{
  if ( (HANDLE) find_ != INVALID_HANDLE_VALUE)
  {
    FindClose ( (HANDLE) find_ );

    find_ = (void*) INVALID_HANDLE_VALUE;
  }
}


//-----------------------------------------------------------------------
//   rewind
//-----------------------------------------------------------------------


void WinDirListing::rewind ()
{
  if ( (HANDLE) find_ != INVALID_HANDLE_VALUE )
  {
    BOOL  result = FindClose ( (HANDLE) find_ );

    find_ = (void*) INVALID_HANDLE_VALUE;

    if ( ! result )
    {
      String  details = NativeSystem::strerror ();

      throw IOException (
        JEM_FUNC,
        String::format (
          "error closing directory `%s\' (%s)",
          name_,
          details
        )
      );
    }
  }
}


//-----------------------------------------------------------------------
//   nextEntry
//-----------------------------------------------------------------------


String WinDirListing::nextEntry ()
{
  WIN32_FIND_DATAA  fdata;
  DWORD             ierr;


  ierr = ERROR_SUCCESS;

  if ( (HANDLE) find_ == INVALID_HANDLE_VALUE )
  {
    // Start a new directory traversal.

    char   path[MAX_PATH + 1];
    idx_t  len = 0;

    if ( name_.size() > 0 && name_ != "." )
    {
      len = name_.size ();

      std::memcpy ( path, name_.addr(), len );

      if ( path[len - 1] != '\\' )
      {
        path[len++] = '\\';
      }
    }

    path[len++] = '*';
    path[len++] = '\0';

    find_ = (void*) FindFirstFileA ( path, &fdata );

    if ( (HANDLE) find_ == INVALID_HANDLE_VALUE )
    {
      ierr = GetLastError ();
    }
  }
  else
  {
    if ( ! FindNextFileA( (HANDLE) find_, &fdata ) )
    {
      ierr = GetLastError ();
    }
  }

  if ( ierr == ERROR_SUCCESS )
  {
    return jem::clone ( fdata.cFileName );
  }

  if ( ierr == ERROR_NO_MORE_FILES )
  {
    rewind ();
  }
  else
  {
    String  details = NativeSystem::strerror ();

    throw IOException (
      JEM_FUNC,
      String::format (
        "error reading directory `%s\' (%s)",
        name_,
        details
      )
    );
  }

  return String ();
}


JEM_END_PACKAGE( io )
