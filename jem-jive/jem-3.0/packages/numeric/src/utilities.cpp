
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


#include <cmath>
#include <jem/base/limits.h>
#include <jem/base/Thread.h>
#include <jem/base/CancelledException.h>
#include <jem/numeric/utilities.h>


JEM_BEGIN_PACKAGE( numeric )


//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


static const lint  CHECK_FLOP_COUNT = 100000_lint;


//-----------------------------------------------------------------------
//   testCancelled
//-----------------------------------------------------------------------


bool testCancelled ( lint& flopCount )
{
  if ( flopCount > CHECK_FLOP_COUNT )
  {
    flopCount = 0;
    return Thread::cancelled ();
  }
  else
  {
    return false;
  }
}


void testCancelled

  ( lint&        flopCount,
    const char*  where,
    const char*  what )

{
  if ( flopCount > CHECK_FLOP_COUNT )
  {
    if ( Thread::cancelled() )
    {
      throw CancelledException ( where, what );
    }

    flopCount = 0;
  }
}


//-----------------------------------------------------------------------
//   modulus
//-----------------------------------------------------------------------


double modulus ( double x, double y )
{
  double  u = abs ( x );
  double  v = abs ( y );
  double  w;

  if ( isTiny( u + v ) )
  {
    return 0.0;
  }
  else if ( u >= v )
  {
    w = y / x;
    return ( u * std::sqrt( 1.0 + w * w ) );
  }
  else
  {
    w = x / y;
    return ( v * std::sqrt( 1.0 + w * w ) );
  }
}


JEM_END_PACKAGE( numeric )
