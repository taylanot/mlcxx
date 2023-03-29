
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


#include <jem/util/Pair.h>
#include "error.h"
#include "Optable.h"


JEM_BEGIN_PACKAGE   ( mp )
JEM_BEGIN_NAMESPACE ( mt )


//=======================================================================
//   class SumOp
//=======================================================================


class SumOp
{
 public:

  template <class T>
    static inline void apply ( T& x, const T& y )
  {
    x = (T) (x + y);
  }

};


//=======================================================================
//   class ProductOp
//=======================================================================


class ProductOp
{
 public:

  template <class T>
    static inline void apply ( T& x, const T& y )
  {
    x = (T) (x * y);
  }

};


//=======================================================================
//   class MinOp
//=======================================================================


class MinOp
{
 public:

  template <class T>
    static inline void apply ( T& x, const T& y )
  {
    if ( y < x )
    {
      x = y;
    }
  }

  static inline void   apply ( IntPair& x, const IntPair& y )
  {
    if ( y.first < x.first )
    {
      x = y;
    }
  }

  static inline void   apply ( DoubleInt& x, const DoubleInt& y )
  {
    if ( y.first < x.first )
    {
      x = y;
    }
  }

};


//=======================================================================
//   class MaxOp
//=======================================================================


class MaxOp
{
 public:

  template <class T>
    static inline void apply ( T& x, const T& y )
  {
    if ( x < y )
    {
      x = y;
    }
  }

  static inline void   apply ( IntPair& x, const IntPair& y )
  {
    if ( x.first < y.first )
    {
      x = y;
    }
  }

  static inline void   apply ( DoubleInt& x, const DoubleInt& y )
  {
    if ( x.first < y.first )
    {
      x = y;
    }
  }

};


//=======================================================================
//   class BorOp
//=======================================================================


class BorOp
{
 public:

  template <class T>
    static void inline apply ( T& x, const T& y )
  {
    x = (T) (x | y);
  }

};


//=======================================================================
//   class BandOp
//=======================================================================


class BandOp
{
 public:

  template <class T>
    static void inline apply ( T& x, const T& y )
  {
    x = (T) (x & y);
  }

};


//=======================================================================
//   class LorOp
//=======================================================================


class LorOp
{
 public:

  template <class T>
    static void inline apply ( T& x, const T& y )
  {
    x = (T) (x || y);
  }

};


//=======================================================================
//   class LandOp
//=======================================================================


class LandOp
{
 public:

  template <class T>
    static void inline apply ( T& x, const T& y )
  {
    x = (T) (x && y);
  }

};


//=======================================================================
//   class ReduceHandler
//=======================================================================


template <class T, class Op>

  class ReduceHandler

{
 public:

  static int  reduce ( const Buffer&  rbuf,
                       const Buffer&  sbuf );

};


template <class T, class Op>

  int ReduceHandler<T,Op>::reduce

  ( const Buffer&  rbuf,
    const Buffer&  sbuf )

{
  const T*     src  = (const T*) sbuf.addr ();
  T*           dst  = (T*)       rbuf.addr ();
  const idx_t  n    =            rbuf.size ();


#ifndef NDEBUG

  const Type  type = getType ( (T*) 0 );


  if      ( rbuf.size() != sbuf.size() )
  {
    return BUFFER_SIZE_ERROR;
  }
  else if ( rbuf.type() != type )
  {
    return BUFFER_TYPE_ERROR;
  }
  else if ( sbuf.type() != type )
  {
    return BUFFER_TYPE_ERROR;
  }

#endif

  switch ( n )
  {
  case 0:

    break;

  case 1:

    Op::apply ( dst[0], src[0] );

    break;

  case 2:

    Op::apply ( dst[0], src[0] );
    Op::apply ( dst[1], src[1] );

    break;

  case 3:

    Op::apply ( dst[0], src[0] );
    Op::apply ( dst[1], src[1] );
    Op::apply ( dst[2], src[2] );

    break;

  case 4:

    Op::apply ( dst[0], src[0] );
    Op::apply ( dst[1], src[1] );
    Op::apply ( dst[2], src[2] );
    Op::apply ( dst[3], src[3] );

    break;

  default:

    for ( idx_t i = 0; i < n; i++ )
    {
      Op::apply ( dst[i], src[i] );
    }
  }

  return 0;
}


//=======================================================================
//   class Optable::Utils_
//=======================================================================


class Optable::Utils_
{
 public:

  template <class T>

    static void   initIntegralHandlers ( T*  type );

  template <class T>

    static void   initFloatHandlers    ( T*  type );

  template <class T>

    static void   initPairHandlers     ( T*  type );

};


//-----------------------------------------------------------------------
//   initIntegralHandlers
//-----------------------------------------------------------------------


template <class T>

  void Optable::Utils_::initIntegralHandlers ( T* type )

{
  const Type  t = getType ( type );

  handlers_[SUM ][t] = & ReduceHandler<T,SumOp    > :: reduce;
  handlers_[PROD][t] = & ReduceHandler<T,ProductOp> :: reduce;
  handlers_[MIN ][t] = & ReduceHandler<T,MinOp    > :: reduce;
  handlers_[MAX ][t] = & ReduceHandler<T,MaxOp    > :: reduce;
  handlers_[BOR ][t] = & ReduceHandler<T,BorOp    > :: reduce;
  handlers_[BAND][t] = & ReduceHandler<T,BandOp   > :: reduce;
  handlers_[LOR ][t] = & ReduceHandler<T,LorOp    > :: reduce;
  handlers_[LAND][t] = & ReduceHandler<T,LandOp   > :: reduce;
}


//-----------------------------------------------------------------------
//   initFloatHandlers
//-----------------------------------------------------------------------


template <class T>

  void Optable::Utils_::initFloatHandlers ( T* type )

{
  const Type  t = getType ( type );

  handlers_[SUM ][t] = & ReduceHandler<T,SumOp    > :: reduce;
  handlers_[PROD][t] = & ReduceHandler<T,ProductOp> :: reduce;
  handlers_[MIN ][t] = & ReduceHandler<T,MinOp    > :: reduce;
  handlers_[MAX ][t] = & ReduceHandler<T,MaxOp    > :: reduce;
}


//-----------------------------------------------------------------------
//   initPairHandlers
//-----------------------------------------------------------------------


template <class T>

  void Optable::Utils_::initPairHandlers ( T* type )

{
  const Type  t = getType ( type );

  handlers_[MIN][t] = & ReduceHandler<T,MinOp>::reduce;
  handlers_[MAX][t] = & ReduceHandler<T,MaxOp>::reduce;
}


//=======================================================================
//   class Optable
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


Optable::Handler Optable::handlers_[OPCODE_COUNT][TYPE_COUNT];


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


void Optable::init ()
{
  for ( int i = 0; i < OPCODE_COUNT; i++ )
  {
    for ( int j = 0; j < TYPE_COUNT; j++ )
    {
      handlers_[i][j] = (Handler) 0;
    }
  }

  Utils_::initIntegralHandlers ( (byte*)      nullptr );
  Utils_::initIntegralHandlers ( (char*)      nullptr );
  Utils_::initIntegralHandlers ( (short*)     nullptr );
  Utils_::initIntegralHandlers ( (int*)       nullptr );
  Utils_::initIntegralHandlers ( (long*)      nullptr );
  Utils_::initIntegralHandlers ( (lint*)      nullptr );
  Utils_::initIntegralHandlers ( (idx_t*)     nullptr );
  Utils_::initFloatHandlers    ( (float*)     nullptr );
  Utils_::initFloatHandlers    ( (double*)    nullptr );
  Utils_::initPairHandlers     ( (IntPair*)   nullptr );
  Utils_::initPairHandlers     ( (DoubleInt*) nullptr );
}


//-----------------------------------------------------------------------
//   dummyHandler
//-----------------------------------------------------------------------


int Optable::dummyHandler

  ( const Buffer&  recvBuf,
    const Buffer&  sendBuf )

{
  return 0;
}


JEM_END_NAMESPACE ( mt )
JEM_END_PACKAGE   ( mp )
