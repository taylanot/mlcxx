
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


#include <jem/base/CString.h>
#include <jem/base/native/System.h>
#include <jem/io/IOException.h>
#include "native/PosixDirListing.h"

extern "C"
{
  #include <errno.h>
  #include <dirent.h>
}


JEM_BEGIN_PACKAGE( io )


//=======================================================================
//   class PosixDirListing
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


PosixDirListing::PosixDirListing ( const String& name ) :

  dir_  ( nullptr ),
  name_ ( name )

{
  dir_ = ::opendir ( makeCString( name ) );

  if ( ! dir_ )
  {
    String  details = NativeSystem::strerror ();

    throw IOException (
      JEM_FUNC,
      String::format (
        "error opening directory `%s\' (%s)",
        name,
        details
      )
    );
  }
}


PosixDirListing::~PosixDirListing ()
{
  if ( dir_ )
  {
    ::closedir ( (DIR*) dir_ );

    dir_ = nullptr;
  }
}


//-----------------------------------------------------------------------
//   rewind
//-----------------------------------------------------------------------


void PosixDirListing::rewind ()
{
  ::rewinddir ( (DIR*) dir_ );
}


//-----------------------------------------------------------------------
//   nextEntry
//-----------------------------------------------------------------------


String PosixDirListing::nextEntry ()
{
  struct dirent*  entry;

  errno = 0;
  entry = ::readdir ( (DIR*) dir_ );

  if ( entry )
  {
    return jem::clone ( entry->d_name );
  }

  if ( errno )
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
