
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


#include <jem/mp/config.h>
#include <jem/mp/MPException.h>

#ifdef JEM_USE_MPI

#include <jem/base/limits.h>
#include <jem/mp/mpi.h>
#include <jem/mp/Type.h>
#include "error.h"


JEM_BEGIN_PACKAGE   ( mp )
JEM_BEGIN_NAMESPACE ( mpi )


//-----------------------------------------------------------------------
//   makeErrorString
//-----------------------------------------------------------------------


String makeErrorString ( int err ) noexcept
{
  char  str[MPI_MAX_ERROR_STRING];
  int   n;

  MPI_Error_string ( err, str, & n );

  return String ( str, str + n );
}


//-----------------------------------------------------------------------
//   raiseError
//-----------------------------------------------------------------------


void raiseError ( const String& context, int err )
{
  if ( err )
  {
    throw MPException ( context, makeErrorString( err ) );
  }
}


//-----------------------------------------------------------------------
//   bufferTypeError
//-----------------------------------------------------------------------


void bufferTypeError

  ( const String&  context,
    int            recvType,
    int            sendType )

{
  throw MPException (
    context,
    String::format (
      "buffer type mismatch: recv buf has type `%s\' "
      "while send buf has type `%s\'",
      typeName ( recvType ),
      typeName ( sendType )
    )
  );
}


//-----------------------------------------------------------------------
//   bufferSizeError
//-----------------------------------------------------------------------


void             bufferSizeError

  ( const String&  context,
    int            recvSize,
    int            sendSize )

{
  throw MPException (
    context,
    String::format (
      "buffer size mismatch: recv buf has size %d "
      "while send buf has size %d",
      recvSize,
      sendSize
    )
  );
}


//-----------------------------------------------------------------------
//   sizeOverflowError
//-----------------------------------------------------------------------


void             sizeOverflowError

  ( const String&  context,
    idx_t          size )

{
  throw MPException (
    context,
    String::format (
      "buffer size (%d) too large; "
      "maximum size supported by MPI is %d",
      size,
      maxOf<int>()
    )
  );
}


JEM_END_NAMESPACE ( mpi )
JEM_END_PACKAGE   ( mp )

#endif
