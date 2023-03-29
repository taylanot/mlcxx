
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

#ifndef JEM_NUMERIC_UTILITIES_H
#define JEM_NUMERIC_UTILITIES_H

#include <cstdlib>
#include <cmath>
#include <jem/defines.h>



JEM_BEGIN_PACKAGE( numeric )


//-----------------------------------------------------------------------
//   testCancelled
//-----------------------------------------------------------------------


bool              testCancelled

  ( lint&           flopCount );

void              testCancelled

  ( lint&           flopCount,
    const char*     where,
    const char*     what );


//-----------------------------------------------------------------------
//   abs
//-----------------------------------------------------------------------


inline short      abs  ( short  s ) noexcept;
inline int        abs  ( int    i ) noexcept;
#ifdef JEM_LINT_IS_LONG_LONG
inline long       abs  ( long   l ) noexcept;
#endif
inline lint       abs  ( lint   l ) noexcept;
inline float      abs  ( float  f ) noexcept;
inline double     abs  ( double d ) noexcept;


//-----------------------------------------------------------------------
//   sqrt
//-----------------------------------------------------------------------


inline float      sqrt ( float  f );
inline double     sqrt ( double d );


//-----------------------------------------------------------------------
//   filter classes
//-----------------------------------------------------------------------


class PassFilter     {};
class DiagonalFilter {};
class BandFilter     {};
class GreaterFilter  {};
class EqualFilter    {};


//-----------------------------------------------------------------------
//   utility functions
//-----------------------------------------------------------------------


template <class T>

  T               modulus

  ( const T&        a,
    const T&        b );

inline float      modulus

  ( float           a,
    float           b );

double            modulus

  ( double          a,
    double          b );

template <class T>

  inline T        sign

  ( const T&        a,
    const T&        b );

template <class T>

  inline void     addToSum

  ( T&              sum,
    T&              err,
    const T&        val );

template <class T>

  inline void     closeSum

  ( T&              sum,
    const T&        err );

template <class T>

  inline void     closeSum

  ( T&              sum,
    const T&        err,
    const T&        val );




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   abs
//-----------------------------------------------------------------------


inline short abs ( short s )   noexcept
{
  return (short) std::abs ( s );
}

inline int abs ( int i )       noexcept
{
  return std::abs ( i );
}

inline long abs ( long l )     noexcept
{
  return std::labs ( l );
}

inline lint abs ( lint l )     noexcept
{
  return std::llabs ( l );
}

inline idx_t abs ( idx_t i )   noexcept
{
  return idx_t ( abs( i.value ) );
}

inline float abs ( float f )   noexcept
{
  return (float) std::fabs ( f );
}


inline double abs ( double d ) noexcept
{
  return std::fabs ( d );
}


//-----------------------------------------------------------------------
//   sqrt
//-----------------------------------------------------------------------


inline float sqrt ( float f )
{
  return (float) std::sqrt ( (double) f );
}


inline double sqrt ( double d )
{
  return std::sqrt ( d );
}


//-----------------------------------------------------------------------
//   modulus
//-----------------------------------------------------------------------


template <class T>

  T modulus ( const T& x, const T& y )

{
  T  u = abs ( x );
  T  v = abs ( y );
  T  s;

  if      ( v < u )
  {
    s = u;
  }
  else if ( u < v )
  {
    s = v;
  }
  else
  {
    return T ();
  }

  x /= s;
  y /= s;

  return ( s * sqrt( x * x + y * y ) );
}


inline float modulus ( float x, float y )
{
  return (float) modulus ( (double) x, (double) y );
}


//-----------------------------------------------------------------------
//   sign
//-----------------------------------------------------------------------


template <class T>

  inline T  sign ( const T& a, const T& b )

{
  return (b < (T) 0 ? -abs( a ) : abs( a ));
}


//-----------------------------------------------------------------------
//   addToSum
//-----------------------------------------------------------------------


template <class T>

  inline void     addToSum

  ( T&              sum,
    T&              err,
    const T&        val )

{
  T  t = val - err;
  T  u = sum + t;

  err  = (u - sum) - t;
  sum  =  u;
}


//-----------------------------------------------------------------------
//   closeSum
//-----------------------------------------------------------------------


template <class T>

  inline void     closeSum

  ( T&              sum,
    const T&        err )

{
  sum = sum - err;
}


template <class T>

  inline void     closeSum

  ( T&              sum,
    const T&        err,
    const T&        val )

{
  sum = sum + (val - err);
}


JEM_END_PACKAGE( numeric )

#endif
