
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
#include <jem/gl/utilities.h>


JEM_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   constants
//-----------------------------------------------------------------------


const GLdouble  RADS_PER_DEGREE =

  (GLdouble) (3.14159265358979323846 / 180.0);

const GLdouble  DEGREES_PER_RAD =

  (GLdouble) (180.0 / 3.14159265358979323846);


//-----------------------------------------------------------------------
//   adjustTickCount
//-----------------------------------------------------------------------


void adjustTickCount ( int& n ) noexcept
{
  if      ( n < 2 )
  {
    n = 0;
  }
  else if ( n < 3 )
  {
    n = 2;
  }
  else if ( n < 5 )
  {
    n = 3;
  }
  else if ( n < 6 )
  {
    n = 5;
  }
  else if ( n < 11 )
  {
    n = 6;
  }
  else if ( n < 21 )
  {
    n = 11;
  }
  else
  {
    n = 21;
  }
}


//-----------------------------------------------------------------------
//   round
//-----------------------------------------------------------------------


void        round

  ( float&    val,
    float     step ) noexcept

{
  double  v = val;
  double  s = step;

  if ( v < 0.0 )
  {
    v = std::floor ( (v + 0.5 * s) / s );
  }
  else
  {
    v = std::ceil  ( (v - 0.5 * s) / s );
  }

  val = (float) (v * s);
}


void        round

  ( double&   val,
    double    step ) noexcept

{
  if ( val < 0.0 )
  {
    val = std::floor ( (val + 0.5 * step) / step );
  }
  else
  {
    val = std::ceil  ( (val - 0.5 * step) / step );
  }

  val *= step;
}


JEM_END_PACKAGE( gl )
