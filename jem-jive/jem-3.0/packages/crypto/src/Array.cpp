
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


#include <jem/base/StringBuffer.h>
#include <jem/crypto/Array.h>


JEM_BEGIN_PACKAGE( crypto )


//=======================================================================
//   utility functions
//=======================================================================

//-----------------------------------------------------------------------
//   toString
//-----------------------------------------------------------------------


String toString ( const ByteVector& value )
{
  const char*   digits = "0123456789abcdef";
  const idx_t   count  = value.size ();

  StringBuffer  buf;


  buf.reserve ( 2 * count );

  for ( idx_t i = 0; i < count; i++ )
  {
    byte  b = value[i];

    if ( b < 16 )
    {
      buf += '0';
    }
    else
    {
      buf += digits[b / 16];
      b    = b % 16;
    }

    buf += digits[b];
  }

  return buf.toString ();
}


JEM_END_PACKAGE( crypto )

