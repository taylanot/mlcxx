
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
#include <cstdlib>
#include <jem/io/utilities.h>
#include <jem/io/PrintWriter.h>
#include <jem/io/StdoutWriter.h>
#include <jem/util/Flex.h>
#include <jem/util/Properties.h>


using namespace jem;
using namespace jem::util;

using jem::io::getWidthOf;


//=======================================================================
//   private functions
//=======================================================================

//-----------------------------------------------------------------------
//   dprint_ ( const bool*, idx_t )
//-----------------------------------------------------------------------


static void dprint_ ( const bool* f, idx_t n )
{
  if ( n < 21 )
  {
    std::printf ( "[ %d", (int) f[0] );

    for ( idx_t i = 1; i < n; i++ )
    {
      std::printf ( ", %d", (int) f[i] );
    }

    std::puts ( " ]" );
  }
  else
  {
    int  w = getWidthOf ( n );

    for ( idx_t i = 0; i < n; i++ )
    {
      std::printf ( "  %*ld : ", w, (long) i );

      if ( f[i] )
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
//   dprint_ ( const char*, idx_t )
//-----------------------------------------------------------------------


static void dprint_ ( const char* f, idx_t n )
{
  for ( idx_t i = 0; i < n; i++ )
  {
    std::putchar ( f[i] );
  }

  std::putchar ( '\n' );
}


//-----------------------------------------------------------------------
//   dprint_ ( const int*, idx_t )
//-----------------------------------------------------------------------


static void dprint_ ( const int* f, idx_t n )
{
  if ( n < 9 )
  {
    std::printf ( "[ %d", f[0] );

    for ( idx_t i = 1; i < n; i++ )
    {
      std::printf ( ", %d", f[i] );
    }

    std::puts ( " ]" );
  }
  else
  {
    int  w1, w2, w3;

    w2 = f[0];
    w3 = 0;

    for ( idx_t i = 1; i < n; i++ )
    {
      w1 = std::abs ( f[i] );

      if ( w1 > w2 )
      {
        w2 = w1;
      }

      if ( f[i] < 0 )
      {
        w3 = 1;
      }
    }

    w1 =      getWidthOf ( n  );
    w2 = w3 + getWidthOf ( w2 );

    for ( idx_t i = 0; i < n; i++ )
    {
      std::printf ( "  %*ld : %*d\n", w1, (long) i, w2, f[i] );
    }
  }
}


//-----------------------------------------------------------------------
//   dprint_ ( const double*, idx_t )
//-----------------------------------------------------------------------


static void dprint_ ( const double* f, idx_t n )
{
  if ( n < 9 )
  {
    std::printf ( "[ %e", f[0] );

    for ( idx_t i = 1; i < n; i++ )
    {
      std::printf ( ", %e", f[i] );
    }

    std::puts ( " ]" );
  }
  else
  {
    int  w = getWidthOf ( n );

    for ( idx_t i = 0; i < n; i++ )
    {
      std::printf ( "  %*ld : %+e\n", w, (long) i, f[i] );
    }
  }
}


//-----------------------------------------------------------------------
//   dprint_ ( Flex<T> )
//-----------------------------------------------------------------------


template <class T>

  static inline void dprint_ ( const Flex<T>& f )

{
  if ( f.size() == 0 )
  {
    std::puts ( "[]" );
  }
  else
  {
    dprint_ ( f.addr(), f.size() );
  }
}


//-----------------------------------------------------------------------
//   dprint_ ( Properties )
//-----------------------------------------------------------------------


static void dprint_ ( const Properties& props )
{
  using jem::io::PrintWriter;
  using jem::io::StdoutWriter;

  Ref<PrintWriter>  prn =

    newInstance<PrintWriter> ( newInstance<StdoutWriter>() );

  prn->incrIndentLevel ();
  props.printTo        ( *prn );
  prn->flush           ();
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
//   dprint ( Flex<bool> )
//-----------------------------------------------------------------------


extern void dprint ( const Flex<bool>* f )
{
  dprintp_ ( f );
}


//-----------------------------------------------------------------------
//   dprint ( Flex<char> )
//-----------------------------------------------------------------------


extern void dprint ( const Flex<char>* f )
{
  dprintp_ ( f );
}


//-----------------------------------------------------------------------
//   dprint ( Flex<int> )
//-----------------------------------------------------------------------


extern void dprint ( const Flex<int>* f )
{
  dprintp_ ( f );
}


//-----------------------------------------------------------------------
//   dprint ( Flex<double> )
//-----------------------------------------------------------------------


extern void dprint ( const Flex<double>* f )
{
  dprintp_ ( f );
}


//-----------------------------------------------------------------------
//   dprint ( Properties )
//-----------------------------------------------------------------------


extern void dprint ( const Properties* props )
{
  dprintp_ ( props );
}
