
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
 *  This example illustrates the differences between the classes
 *  StandardAllocator, MallocAllocator and FastAllocator.
 *
 ***********************************************************************/

#include <jem/base/limits.h>
#include <jem/base/System.h>
#include <jem/util/Timer.h>
#include <jem/util/FastAllocator.h>
#include <jem/util/StdAllocator.h>
#include <jem/util/MallocAllocator.h>

#include <new>


using namespace jem;
using namespace jem::io;
using namespace jem::util;


//=======================================================================
//   class Stack
//=======================================================================


template <class T, class Alloc = StdAllocator>

  class Stack

{
 public:

                  Stack ();
                 ~Stack ();

  inline void     push

    ( const T&      val );

  void            clear

    ( int           n = maxOf<int>() );


 private:

  struct          Node_
  {
    inline        Node_

      ( Node_*      nxt,
        const T&    val );


    Node_*        next;
    T             value;
  };

  Node_*          top_;
  Alloc           allocator_;

};


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


template <class T, class Alloc>

  Stack<T,Alloc>::Stack() :

    top_       ( 0 ),
    allocator_ ( sizeof(Node_) )

{}


template <class T, class Alloc>

  Stack<T,Alloc>::~Stack ()

{
  clear ();
}


//-----------------------------------------------------------------------
//   push
//-----------------------------------------------------------------------


template <class T, class Alloc>

  inline void Stack<T,Alloc>::push ( const T& val )

{
  Node_* n = (Node_*) allocator_.alloc ();

  new (n) Node_ ( top_, val );

  top_ = n;
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


template <class T, class Alloc>

  void Stack<T,Alloc>::clear ( int k )

{
  Node_* n;

  while ( top_ && k > 0 )
  {
    n    = top_;
    top_ = top_->next;

    n->~Node_ ();

    allocator_.dealloc ( n );
    k--;
  }
}


//=======================================================================
//   class Stack::Node_
//=======================================================================

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


template <class T, class Alloc>

  inline Stack<T,Alloc>::Node_::Node_

  ( Node_*    nxt,
    const T&  val ) :

    next  ( nxt ),
    value ( val )

{}


//=======================================================================
//   global functions
//=======================================================================

//-----------------------------------------------------------------------
//   pushInts
//-----------------------------------------------------------------------


template <class Alloc>

  void               pushInts

  ( Stack<int,Alloc>&  stack,
    int                n )

{
  for ( int i = 0; i < n; i++ )
  {
    stack.push ( i );
  }
}


//-----------------------------------------------------------------------
//   perftest
//-----------------------------------------------------------------------


template <class Alloc>

  void                perftest

  ( Stack<int,Alloc>&   s,
    const String&       allocName )

{
  const int  N   = 1000000;

  Writer&    out = System::out ();

  Timer      t;


  print ( out, "Testing ", allocName, " ... ", flush );

  t.start ();

  pushInts ( s, N );
  s.clear  ( N / 2 );
  pushInts ( s, N / 2 );
  s.clear  ( 2 * N / 3 );
  pushInts ( s, N / 3 );
  s.clear  ();

  print ( out, "ready in ", t.toDouble(), " seconds\n" );
}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


int run ()
{
  Stack<int,FastAllocator  >   s1;
  Stack<int,StdAllocator   >   s2;
  Stack<int,MallocAllocator>   s3;


  print ( System::out(), '\n' );

  perftest ( s1, "FastAllocator  " );
  perftest ( s2, "StdAllocator   " );
  perftest ( s3, "MallocAllocator" );

  print ( System::out(), '\n' );

  return 0;
}


//-----------------------------------------------------------------------
//   main
//-----------------------------------------------------------------------


int main ()
{
  return System::exec ( & run );
}


