
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


#include <jem/base/CancelledException.h>
#include <jem/mp/MPException.h>
#include "error.h"


JEM_BEGIN_PACKAGE   ( mp )
JEM_BEGIN_NAMESPACE ( mt )


//-----------------------------------------------------------------------
//   error constants
//-----------------------------------------------------------------------


const int  CANCELLED_ERROR     = 1 << 0;
const int  BUFFER_TYPE_ERROR   = 1 << 1;
const int  BUFFER_SIZE_ERROR   = 1 << 2;
const int  REDUCE_TYPE_ERROR   = 1 << 3;
const int  REDUCE_ACTION_ERROR = 1 << 4;
const int  ERROR_COUNT         = 5;


//-----------------------------------------------------------------------
//   makeErrorString
//-----------------------------------------------------------------------


String makeErrorString ( int err ) noexcept
{
  const char*  errors[5] = { "thread cancelled",
                             "buffer type mismatch",
                             "buffer size mismatch",
                             "invalid reduce type",
                             "mixed collective operations" };

  String  str;
  int     e = 1;


  for ( int i = 0; i < ERROR_COUNT; i++ )
  {
    if ( err & e )
    {
      if ( str.size() > 0 )
      {
        str = String::format ( "%s and %s", str, errors[i] );
      }
      else
      {
        str = errors[i];
      }
    }

    e = e << 1;
  }

  if ( str.size() == 0 )
  {
    if ( err )
    {
      str = "unknown error";
    }
    else
    {
      str = "no error";
    }
  }

  return str;
}


//-----------------------------------------------------------------------
//   raiseError
//-----------------------------------------------------------------------


void raiseError ( const String& context, int err )
{
  if ( err )
  {
    if ( err & CANCELLED_ERROR )
    {
      throw CancelledException ( JEM_FUNC );
    }
    else
    {
      throw MPException ( context, makeErrorString( err ) );
    }
  }
}


JEM_END_NAMESPACE ( mt )
JEM_END_PACKAGE   ( mp )
