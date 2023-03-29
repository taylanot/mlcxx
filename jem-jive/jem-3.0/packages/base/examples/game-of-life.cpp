
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

/************************************************************************
 *
 *         This implementation of "the game of life" uses Arrays
 *
 *
 *                    SHORT DESCRIPTION OF THE GAME
 *
 *  In this example a simple game called "Life" will be implemented.
 *  The Game of Life (or simply Life) is not a game in the conventional
 *  sense. There are no players, and no winning or losing.
 *  Life is played on a grid of square cells--like a chess board
 *  but extending infinitely in every direction. A cell can be alive
 *  or dead. Each cell in the grid has a neighborhood consisting of
 *  the eight cells in every direction including diagonals. The game
 *  starts with a (pre-defined) pattern of living cells.
 *
 *  One step in the game of Life is
 *  as follows:
 *
 *     - count the number of neighbours for each cell individually
 *
 *  then change the value of each cell according to the following
 *  rules:
 *
 *     - a dead cell with exactly three living neighbours becomes
 *       alive (birth);
 *
 *     - a living cell with two or three living neighbours stays
 *       alive (survival);
 *
 *     - in all other cases, a cell dies or remains dead
 *       (overcrowding or loneliness);
 *
 *  By repeating this step a life-like evolution behaviour can be
 *  observed.
 *
 *  Note: it is important that the change of each cell is detected
 *  first before any cell is actually changed.
 *
 *  More information about the game and its rules can be found on
 *  the web-site : www.math.com/students/wonders/life/life.html
 *
 *
 *
 *                     GENERAL IMPLEMENTATION REMARKS
 *
 *  The game of Life has been implemented using Arrays,
 *  Array slices and Array expressions. The game starts with a two
 *  dimensional integer Array filled with a pre-defined pattern of
 *  zero and unity numbers. This Array represents the spatial grid
 *  described above. A unity value in the array represents a living
 *  cell while a zero value indicates a dead cell. One evolution step
 *  in the Life algorithm requires the evaluation of the number of
 *  living neighbour cells for each cell in the grid. The counting
 *  is performed as follows:
 *
 *  Let Array<int,2> a represent a two dimensional game-grid of size
 *  NxN and let Array<int,2> b refer to the inner part of that grid
 *  by performing the operation
 *  b.ref( a( slice( 1, N - 1  ), slice( 1, N - 1 ) ).
 *  In a similar way another Array c can be created by
 *  c.ref( a( slice( 0, N - 2 ), slice( 0, N - 2 ) ). Now notice
 *  that c(i,j) refers to the left ( or west-side ) neighbor of the
 *  cell b(i,j). By creating 8 Arrays each referring to another
 *  neighbor, the number of living neighbours for each cell can
 *  be obtained by a simple summation of all t neighbor Arrays.
 *
 ***********************************************************************/

#include <jem/base/Time.h>
#include <jem/base/Array.h>
#include <jem/base/System.h>
#include <jem/base/Thread.h>
#include <jem/io/FileWriter.h>
#include <jem/io/StdoutWriter.h>
#include <jem/io/BufferedWriter.h>


using namespace jem;
using namespace jem::io;
using namespace jem::util;


//-----------------------------------------------------------------------
//   printGame
//-----------------------------------------------------------------------


void                    printGame

  ( TextOutput&           out,
    const Array<int,2>&   g )

{
  print ( out, " +-" );

  for ( idx_t i = 0; i < g.size(0); i++ )
  {
    print ( out, '-' );
  }

  print ( out, "-+", endl );

  for ( idx_t j = 0; j < g.size(1); j++ )
  {
    print ( out, " | " );

    for ( idx_t i = 0; i < g.size(0); i++ )
    {
      if( g(i,j) > 0 )
      {
        print ( out, 'O' );
      }
      else
      {
        print ( out, ' ' );
      }
    }

    print ( out, " |", endl );
  }

  print ( out, " +-" );

  for ( idx_t i = 0; i < g.size(0); i++ )
  {
    print ( out, '-' );
  }

  print ( out, "-+", endl, endl );
}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------

int run ()
{
  const int      P = 35;
  const int      Q = P / 2;

  const int      N = P * 2;
  const int      M = N / 2;

  Array<int,2>   a  ( N,     P     );
  Array<int,2>   b  ( N - 2, P - 2 );

  Array<int,2>   c  ( a(slice(1,N - 1),slice(1,P - 1)) );
  Array<int,2>   n  ( a(slice(1,N - 1),slice(0,P - 2)) );
  Array<int,2>   w  ( a(slice(0,N - 2),slice(1,P - 1)) );
  Array<int,2>   e  ( a(slice(2,N),    slice(1,P - 1)) );
  Array<int,2>   s  ( a(slice(1,N - 1),slice(2,P    )) );
  Array<int,2>   nw ( a(slice(0,N - 2),slice(0,P - 2)) );
  Array<int,2>   ne ( a(slice(2,N    ),slice(0,P - 2)) );
  Array<int,2>   sw ( a(slice(0,N - 2),slice(2,P    )) );
  Array<int,2>   se ( a(slice(2,N    ),slice(2,P    )) );


  Ref<Writer>     out =

    newInstance<BufferedWriter> (newInstance<StdoutWriter> () );


  a = 0;

  // Initialize the living cells.

  a(M - 21,Q + 3) = a(M - 20,Q + 3) = a(M - 19,Q + 2) =
  a(M - 18,Q + 1) = a(M - 18,Q    ) = a(M - 18,Q - 1) =
  a(M - 19,Q - 2) = a(M - 20,Q - 3) = a(M - 21,Q - 3) = 1;

  a(M - 1, Q + 3) = a(M    , Q + 3) = a(M + 1, Q + 2) =
  a(M + 2, Q + 1) = a(M + 2, Q    ) = a(M + 2, Q - 1) =
  a(M + 1, Q - 2) = a(M    , Q - 3) = a(M - 1, Q - 3) = 1;

  printGame  ( *out, c );

  out->flush ();

  Thread::sleep ( Time( 1.0 ) );

  for ( idx_t i = 0; i < 185; ++i )
  {
    b = 0;
    b = n + s + e + w + ne + nw + se + sw;
    c = where ( b == 3, 1, where( b == 2, c, 0 ) );

    printGame  ( *out, c );

    out->flush ();

    Thread::sleep ( Time(0.15) );
  }

  return 0;
}


//-----------------------------------------------------------------------
//   main
//-----------------------------------------------------------------------


int main ()
{
  return System::exec ( & run );
}
