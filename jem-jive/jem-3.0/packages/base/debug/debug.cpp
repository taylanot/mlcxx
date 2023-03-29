
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


#include <cstdio>
#include <jem/base/Object.h>
#include <jem/base/tuple/Tuple.h>
#include <jem/base/array/Array.h>
#include <jem/base/array/intrinsics.h>
#include <jem/base/array/utilities.h>
#include <jem/io/utilities.h>


using namespace jem;

using jem::io::getWidthOf;


//=======================================================================
//   private functions
//=======================================================================

//-----------------------------------------------------------------------
//   dprint_ ( bool )
//-----------------------------------------------------------------------


static void dprint_ ( bool b )
{
  if ( b )
  {
    std::puts ( "true" );
  }
  else
  {
    std::puts ( "false" );
  }
}


//-----------------------------------------------------------------------
//   dprint_ ( char )
//-----------------------------------------------------------------------


static void dprint_ ( char c )
{
  std::putchar ( c    );
  std::putchar ( '\n' );
}


//-----------------------------------------------------------------------
//   dprint_ ( int )
//-----------------------------------------------------------------------


static void dprint_ ( int i )
{
  std::printf ( "%d\n", i );
}


//-----------------------------------------------------------------------
//   dprint_ ( lint )
//-----------------------------------------------------------------------


static void dprint_ ( lint i )
{
  std::printf ( "%ld\n", (long) i );
}


//-----------------------------------------------------------------------
//   dprint ( float )
//-----------------------------------------------------------------------


static void dprint_ ( float f )
{
  std::printf ( "%e\n", f );
}


//-----------------------------------------------------------------------
//   dprint_ ( double )
//-----------------------------------------------------------------------


static void dprint_ ( double d )
{
  std::printf ( "%e\n", d );
}


//-----------------------------------------------------------------------
//   dprint_ ( const char* )
//-----------------------------------------------------------------------


static void dprint_ ( const char* s )
{
  if ( s )
  {
    std::puts ( s );
  }
  else
  {
    std::puts ( "null" );
  }
}


//-----------------------------------------------------------------------
//   dprint_ ( String )
//-----------------------------------------------------------------------


static void dprint_ ( const String& s )
{
  const idx_t  n = s.size ();

  for ( idx_t i = 0; i < n; i++ )
  {
    std::putchar ( s[i] );
  }

  std::putchar ( '\n' );
}


//-----------------------------------------------------------------------
//   dprint_ ( Object )
//-----------------------------------------------------------------------


static void dprint_ ( const Object& obj )
{
  dprint_ ( obj.toString() );
}


static void dprint_ ( const Ref<Object>& obj )
{
  if  ( obj != NIL )
  {
    dprint_ ( obj->toString() );
  }
  else
  {
    std::puts ( "NIL" );
  }
}


//-----------------------------------------------------------------------
//   dprint_ ( Tuple<int,N> )
//-----------------------------------------------------------------------


template <int N>

  static void dprint_ ( const Tuple<int,N>& t )

{
  std::printf ( "[ %d", t[0] );

  for ( int i = 1; i < N; i++ )
  {
    std::printf ( ", %d", t[i] );
  }

  std::puts ( " ]" );
}


//-----------------------------------------------------------------------
//   dprint_ ( Tuple<double,N> )
//-----------------------------------------------------------------------


template <int N>

  static void dprint_ ( const Tuple<double,N>& t )

{
  std::printf ( "[ %e", t[0] );

  for ( int i = 1; i < N; i++ )
  {
    std::printf ( ", %e", t[i] );
  }

  std::puts ( " ]" );
}


//-----------------------------------------------------------------------
//   dprint_ ( Array<bool> )
//-----------------------------------------------------------------------


static void dprint_ ( const Array<bool>& a )
{
  const idx_t  n = a.size ();

  int          w;
  idx_t        i;


  if ( n == 0 )
  {
    std::puts ( "[]" );
    return;
  }

  if ( n < 21 )
  {
    std::printf ( "[ %d", (int) a[0] );

    for ( i = 1; i < n; i++ )
    {
      std::printf ( ", %d", (int) a[i] );
    }

    std::puts ( " ]" );
  }
  else
  {
    w = getWidthOf ( n );

    for ( i = 0; i < n; i++ )
    {
      std::printf ( "  %*ld : ", w, (long) i );

      if ( a[i] )
      {
        std::printf ( "true\n" );
      }
      else
      {
        std::printf ( "false\n" );
      }
    }
  }
}


//-----------------------------------------------------------------------
//   dprint_ ( Array<bool,2> )
//-----------------------------------------------------------------------


static void dprint_ ( const Array<bool,2>& a )
{
  const idx_t  m = a.size (0);
  const idx_t  n = a.size (1);

  idx_t        i, j;


  if ( m * n == 0 )
  {
    std::puts ( "[]" );
    return;
  }

  for ( i = 0; i < m; i++ )
  {
    std::printf ( "[ %d", (int) a(i,0) );

    for ( j = 1; j < n; j++ )
    {
      std::printf ( ", %d", (int) a(i,j) );
    }

    std::puts ( " ]" );
  }
}


//-----------------------------------------------------------------------
//   dprint_ ( Array<int> )
//-----------------------------------------------------------------------


static void dprint_ ( const Array<int>& a )
{
  const idx_t  n = a.size ();

  int          w1, w2;
  idx_t        i;


  if ( n == 0 )
  {
    std::puts ( "[]" );
    return;
  }

  if ( n < 9 )
  {
    std::printf ( "[ %d", a[0] );

    for ( i = 1; i < n; i++ )
    {
      std::printf ( ", %d", a[i] );
    }

    std::puts ( " ]" );
  }
  else
  {
    w1 = getWidthOf ( n );
    w2 = getWidthOf ( max( abs( a ) ) );

    if ( min( a ) < 0 )
    {
      w2++;
    }

    for ( i = 0; i < n; i++ )
    {
      std::printf ( "  %*ld : %*d\n", w1, (long) i, w2, a[i] );
    }
  }
}


//-----------------------------------------------------------------------
//   dprint_ ( Array<int,2> )
//-----------------------------------------------------------------------


static void dprint_ ( const Array<int,2>& a )
{
  const idx_t  m = a.size (0);
  const idx_t  n = a.size (1);

  int          w;
  idx_t        i, j;


  if ( m * n == 0 )
  {
    std::puts ( "[]" );
    return;
  }

  w = getWidthOf ( max( abs( a ) ) );

  if ( min( a ) < 0 )
  {
    w++;
  }

  for ( i = 0; i < m; i++ )
  {
    std::printf ( "[ %*d", w, a(i,0) );

    for ( j = 1; j < n; j++ )
    {
      std::printf ( ", %*d", w, a(i,j) );
    }

    std::puts ( " ]" );
  }
}


//-----------------------------------------------------------------------
//   dprint_ ( Array<double> )
//-----------------------------------------------------------------------


static void dprint_ ( const Array<double>& a )
{
  const idx_t  n = a.size ();

  int          w;
  idx_t        i;


  if ( n == 0 )
  {
    std::puts ( "[]" );
    return;
  }

  if ( n < 9 )
  {
    std::printf ( "[ %e", a[0] );

    for ( i = 1; i < n; i++ )
    {
      std::printf ( ", %e", a[i] );
    }

    std::puts ( " ]" );
  }
  else
  {
    w = getWidthOf ( n );

    for ( i = 0; i < n; i++ )
    {
      std::printf ( "  %*ld : %+e\n", w, (long) i, a[i] );
    }
  }
}


//-----------------------------------------------------------------------
//   dprint_ ( Array<double,2> )
//-----------------------------------------------------------------------


static void dprint_ ( const Array<double,2>& a )
{
  const idx_t  m = a.size (0);
  const idx_t  n = a.size (1);

  idx_t        i, j;
  int          p;


  if ( m * n == 0 )
  {
    std::puts ( "[]" );
    return;
  }

  if      ( n <= 10 )
  {
    p = 6;
  }
  else if ( n <= 16 )
  {
    p = 4;
  }
  else
  {
    p = 2;
  }

  for ( i = 0; i < m; i++ )
  {
    std::printf ( "[ %+.*e", p, a(i,0) );

    for ( j = 1; j < n; j++ )
    {
      std::printf ( ", %+.*e", p, a(i,j) );
    }

    std::puts ( " ]" );
  }
}


//-----------------------------------------------------------------------
//   dprint_ ( Array<double,3> )
//-----------------------------------------------------------------------


static void dprint_ ( const Array<double,3>& a )
{
  if ( a.size(0) * a.size(1) * a.size(2) == 0 )
  {
    std::puts ( "[]" );
    return;
  }

  dprint_ ( a(ALL,ALL,0) );

  for ( idx_t k = 1; k < a.size(2); k++ )
  {
    std::putchar ( '\n' );
    dprint_      ( a(ALL,ALL,k) );
  }
}


//-----------------------------------------------------------------------
//   dprintp_ ( const T* )
//-----------------------------------------------------------------------


template <class T>

  static inline void dprintp_ ( const T* t )

{
  if ( t )
  {
    dprint_ ( *t );
  }
  else
  {
    std::puts ( "null" );
  }
}


//=======================================================================
//   public functions
//=======================================================================

//-----------------------------------------------------------------------
//   dprint ( bool )
//-----------------------------------------------------------------------


extern void dprint ( const bool* b )
{
  dprintp_ ( b );
}


//-----------------------------------------------------------------------
//   dprint ( char )
//-----------------------------------------------------------------------


extern void dprint ( const char* c )
{
  dprintp_ ( c );
}


//-----------------------------------------------------------------------
//   dprint ( int )
//-----------------------------------------------------------------------


extern void dprint ( const int* i )
{
  dprintp_ ( i );
}


//-----------------------------------------------------------------------
//   dprint ( lint )
//-----------------------------------------------------------------------


extern void dprint ( const lint* i )
{
  dprintp_ ( i );
}


//-----------------------------------------------------------------------
//   dprint ( float )
//-----------------------------------------------------------------------


extern void dprint ( const float* f )
{
  dprintp_ ( f );
}


//-----------------------------------------------------------------------
//   dprint ( double )
//-----------------------------------------------------------------------


extern void dprint ( const double* d )
{
  dprintp_ ( d );
}


//-----------------------------------------------------------------------
//   dprint ( void* )
//-----------------------------------------------------------------------


extern void dprint ( void* p )
{
  if ( p )
  {
    std::printf ( "%p\n", p );
  }
  else
  {
    std::puts   ( "null" );
  }
}


//-----------------------------------------------------------------------
//   dprint ( const char* )
//-----------------------------------------------------------------------


extern void dprint ( const char** s )
{
  dprintp_ ( s );
}


//-----------------------------------------------------------------------
//   dprint ( String )
//-----------------------------------------------------------------------


extern void dprint ( const String* s )
{
  dprintp_ ( s );
}


//-----------------------------------------------------------------------
//   dprint ( Object )
//-----------------------------------------------------------------------


extern void dprint ( const Object* obj )
{
  dprintp_ ( obj );
}


extern void dprint ( const Ref<Object>* obj )
{
  dprintp_ ( obj );
}


//-----------------------------------------------------------------------
//   dprint ( Tuple<int,N> )
//-----------------------------------------------------------------------


extern void dprint ( const Tuple<int,1>* t )
{
  dprintp_ ( t );
}


extern void dprint ( const Tuple<int,2>* t )
{
  dprintp_ ( t );
}


extern void dprint ( const Tuple<int,3>* t )
{
  dprintp_ ( t );
}


extern void dprint ( const Tuple<int,4>* t )
{
  dprintp_ ( t );
}


//-----------------------------------------------------------------------
//   dprint ( Tuple<double,N> )
//-----------------------------------------------------------------------


extern void dprint ( const Tuple<double,1>* t )
{
  dprintp_ ( t );
}


extern void dprint ( const Tuple<double,2>* t )
{
  dprintp_ ( t );
}


extern void dprint ( const Tuple<double,3>* t )
{
  dprintp_ ( t );
}


extern void dprint ( const Tuple<double,4>* t )
{
  dprintp_ ( t );
}


//-----------------------------------------------------------------------
//   dprint ( Array<bool> )
//-----------------------------------------------------------------------


extern void dprint ( const Array<bool>* a )
{
  dprintp_ ( a );
}


//-----------------------------------------------------------------------
//   dprint ( Array<bool,2> )
//-----------------------------------------------------------------------


extern void dprint ( const Array<bool,2>* a )
{
  dprintp_ ( a );
}


//-----------------------------------------------------------------------
//   dprint ( Array<int> )
//-----------------------------------------------------------------------


extern void dprint ( const Array<int>* a )
{
  dprintp_ ( a );
}


//-----------------------------------------------------------------------
//   dprint ( Array<int,2> )
//-----------------------------------------------------------------------


extern void dprint ( const Array<int,2>* a )
{
  dprintp_ ( a );
}


//-----------------------------------------------------------------------
//   dprint ( Array<double> )
//-----------------------------------------------------------------------


extern void dprint ( const Array<double>* a )
{
  dprintp_ ( a );
}


//-----------------------------------------------------------------------
//   dprint ( Array<double,2> )
//-----------------------------------------------------------------------


extern void dprint ( const Array<double,2>* a )
{
  dprintp_ ( a );
}


//-----------------------------------------------------------------------
//   dprint ( Array<double,3> )
//-----------------------------------------------------------------------


extern void dprint ( const Array<double,3>* a )
{
  dprintp_ ( a );
}
