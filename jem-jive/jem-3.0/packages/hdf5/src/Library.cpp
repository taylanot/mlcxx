
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
#include <cstdlib>
#include <jem/base/Once.h>
#include <jem/util/Flex.h>
#include <jem/hdf5/Library.h>
#include "jemhdf5.h"


JEM_BEGIN_PACKAGE( hdf5 )


//=======================================================================
//   private data
//=======================================================================


static const size_t         MAX_ERROR_LEN_ = 256;


//=======================================================================
//   private functions
//=======================================================================

//-----------------------------------------------------------------------
//   getErrorString_
//-----------------------------------------------------------------------


extern "C"
{
  static herr_t             getErrorString_

    ( unsigned int            n,
      const H5E_error2_t*     eptr,
      void*                   data )

  {
    if ( n == 0 )
    {
      char*   buf = (char*) data;
      size_t  len = std::strlen ( eptr->desc );

      if ( len > MAX_ERROR_LEN_ )
      {
        len = MAX_ERROR_LEN_;
      }

      std::memcpy ( buf, eptr->desc, len );

      buf[len] = '\0';
    }

    return 0;
  }
}


//=======================================================================
//   class Library
//=======================================================================

//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


void Library::init ()
{
  static Once  once = JEM_ONCE_INITIALIZER;

  runOnce ( once, init_ );
}


//-----------------------------------------------------------------------
//   strerror
//-----------------------------------------------------------------------


String Library::strerror () noexcept
{
  String  str;
  char*   buf;

  buf = (char*) std::malloc ( MAX_ERROR_LEN_ + 1 );

  if ( ! buf )
  {
    return str;
  }

  buf[0] = '\0';

  H5Ewalk ( H5E_DEFAULT, H5E_WALK_UPWARD,
            getErrorString_, buf );

  str = clone ( buf );

  std::free ( buf );

  return str.stripWhite ();
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void Library::init_ ()
{
  H5open      ();
  H5Eset_auto ( H5E_DEFAULT, nullptr, nullptr );
}


JEM_END_PACKAGE( hdf5 )

