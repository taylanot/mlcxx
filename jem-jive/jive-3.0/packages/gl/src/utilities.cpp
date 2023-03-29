
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jive, an object oriented toolkit for solving
 *  partial differential equations.
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
 *  This file is part of Jive, an object oriented toolkit for
 *  solving partial differential equations.
 *
 *  Jive version: 3.0
 *  Date:         Fri 20 Dec 14:30:12 CET 2019
 */


#include <cmath>
#include <jem/base/Float.h>
#include <jem/base/IllegalInputException.h>
#include <jem/base/array/intrinsics.h>
#include <jem/base/array/utilities.h>
#include <jive/gl/utilities.h>


using jem::Float;
using jem::IllegalInputException;


JIVE_BEGIN_PACKAGE( gl )


//=======================================================================
//   internal functions
//=======================================================================

//-----------------------------------------------------------------------
//   getScale_
//-----------------------------------------------------------------------


static double     getScale_

  ( double          amax,
    double          step )

{
  if ( Float::isNaN( amax ) )
  {
    throw IllegalInputException ( JEM_FUNC, "invalid data: NaN" );
  }

  if ( step >= 1.0e-5 )
  {
    double  x = amax;

    if ( x > 1.0e-20 && x < 1.0e+20 )
    {
      x    = std::log10( x );
      x    = std::pow( 10.0, std::floor( x + 0.5 ) );
      x    = step * x;

      amax = amax / x;
      amax = x * std::ceil( amax );
    }
  }

  if      ( amax < 1.0e-20 )
  {
    amax = 1.0e-20;
  }
  else if ( amax > 1.0e+20 )
  {
    amax = 1.0e+20;
  }

  return (1.0 / amax);
}


//-----------------------------------------------------------------------
//   getScaleAndBias_
//-----------------------------------------------------------------------


static void       getScaleAndBias_

  ( double&         scale,
    double&         bias,
    double          minVal,
    double          maxVal,
    double          step )

{
  double  x, dx;


  if ( Float::isNaN( minVal ) || Float::isNaN( maxVal ) )
  {
    throw IllegalInputException ( JEM_FUNC, "invalid data: NaN" );
  }

  if ( step >= 1.0e-5 )
  {
    x = jem::max ( std::fabs(minVal), std::fabs(maxVal) );

    if ( x > 1.0e-20 && x < 1.0e+20 )
    {
      x = std::log10( x );
      x = std::pow( 10.0, std::floor( x + 0.5 ) );
      x = step * x;

      if ( maxVal >= 0.0 )
      {
        maxVal = maxVal / x;
        maxVal = x * std::ceil( maxVal );
      }
      else
      {
        maxVal = -maxVal / x;
        maxVal = -x * std::floor( maxVal );
      }

      if ( minVal >= 0.0 )
      {
        minVal = minVal / x;
        minVal = x * std::floor( minVal );
      }
      else
      {
        minVal = -minVal / x;
        minVal = -x * std::ceil( minVal );
      }
    }
  }

  dx = maxVal - minVal;

  if      ( dx < 1.0e-20 )
  {
    dx = 1.0e-20;
  }
  else if ( dx > 1.0e+20 )
  {
    dx = 1.0e20;
  }

  scale =     1.0 / dx;
  bias  = -minVal / dx;
}


//=======================================================================
//   public functions
//=======================================================================

//-----------------------------------------------------------------------
//   getScale (Vector)
//-----------------------------------------------------------------------


double            getScale

  ( const Vector&   data,
    double          step )

{
  if ( data.size() )
  {
    return getScale_ ( jem::max( jem::abs( data ) ), step );
  }
  else
  {
    return 1.0;
  }
}


//-----------------------------------------------------------------------
//   getScale (Matrix)
//-----------------------------------------------------------------------


double            getScale

  ( const Matrix&   data,
    double          step )

{
  if ( (data.size(0) * data.size(1)) > 0 )
  {
    return getScale_ ( jem::max( jem::abs( data ) ), step );
  }
  else
  {
    return 1.0;
  }
}


//-----------------------------------------------------------------------
//   getScaleAndBias (vector)
//-----------------------------------------------------------------------


void              getScaleAndBias

  ( double&         scale,
    double&         bias,
    const Vector&   data,
    double          step )

{
  const idx_t  n = data.size ();

  double       minVal;
  double       maxVal;
  double       val;


  if ( n == 0 )
  {
    scale = 1.0;
    bias  = 0.0;
  }
  else
  {
    minVal = maxVal = data[0];

    for ( idx_t i = 1; i < n; i++ )
    {
      val = data[i];

      if      ( val < minVal )
      {
        minVal = val;
      }
      else if ( val > maxVal )
      {
        maxVal = val;
      }
    }

    getScaleAndBias_ ( scale, bias, minVal, maxVal, step );
  }
}


//-----------------------------------------------------------------------
//   getScaleAndBias (Matrix)
//-----------------------------------------------------------------------

void              getScaleAndBias

  ( double&         scale,
    double&         bias,
    const Matrix&   data,
    double          step )

{
  const idx_t  m = data.size (0);
  const idx_t  n = data.size (1);

  double       minVal;
  double       maxVal;
  double       val;


  if ( m * n == 0 )
  {
    scale = 1.0;
    bias  = 0.0;
  }
  else
  {
    minVal = maxVal = data(0,0);

    for ( idx_t j = 0; j < n; j++ )
    {
      for ( idx_t i = 0; i < m; i++ )
      {
        val = data(i,j);

        if      ( val < minVal )
        {
          minVal = val;
        }
        else if ( val > maxVal )
        {
          maxVal = val;
        }
      }
    }

    getScaleAndBias_ ( scale, bias, minVal, maxVal, step );
  }
}


JIVE_END_PACKAGE( gl )
