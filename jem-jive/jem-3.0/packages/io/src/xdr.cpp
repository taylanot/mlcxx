
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


#include <jem/base/Error.h>
#include <jem/io/xdr.h>


JEM_BEGIN_PACKAGE   ( io )
JEM_BEGIN_NAMESPACE ( xdr )


//-----------------------------------------------------------------------
//   getByteOrder
//-----------------------------------------------------------------------


int getByteOrder ()
{
  // TODO: check std::numeric_limits<float>::is_iec559 and
  //             std::numeric_limits<oduble>::is_iec559

  if ( sizeof(float) != XDR_FLOAT_SIZE )
  {
    throw Error (
      JEM_FUNC,
      "float type does not conform to the IEEE-754 standard"
    );
  }

  if ( sizeof(double) != XDR_DOUBLE_SIZE )
  {
    throw Error (
      JEM_FUNC,
      "double type does not conform to the IEEE-754 standard"
    );
  }

  ulint  i = 0x1;
  byte*  p = (byte*) &i;

  if ( *p )
  {
    return LITTLE_ENDIAN;
  }
  else
  {
    return BIG_ENDIAN;
  }
}


JEM_END_NAMESPACE ( xdr )
JEM_END_PACKAGE   ( io )
