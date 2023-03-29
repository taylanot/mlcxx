
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

#ifndef JEM_MP_UTILITIES_H
#define JEM_MP_UTILITIES_H

#include <jem/mp/Buffer.h>
#include <jem/mp/Context.h>


JEM_BEGIN_PACKAGE( mp )


//-----------------------------------------------------------------------
//   utility functions
//-----------------------------------------------------------------------


template <class T>

  inline void           send

  ( Context&              ctx,
    const T&              val,
    int                   rank,
    int                   tag = DEFAULT_TAG );

template <class T>

  inline void           send

  ( Context&              ctx,
    const T*              buf,
    idx_t                 len,
    int                   rank,
    int                   tag = DEFAULT_TAG );

template <class T>

  inline void           recv

  ( Context&              ctx,
    T&                    val,
    int                   rank,
    int                   tag  = DEFAULT_TAG,
    Status*               stat = 0 );

template <class T>

  inline void           recv

  ( Context&              ctx,
    T*                    buf,
    idx_t                 len,
    int                   rank,
    int                   tag  = DEFAULT_TAG,
    Status*               stat = 0 );

template <class T>

  inline T              allreduce

  ( Context&              ctx,
    const T&              val,
    Opcode                op );

template <class T>

  inline T              allmax

  ( Context&              ctx,
    const T&              val );

template <class T>

  inline T              allmin

  ( Context&              ctx,
    const T&              val );

template <class T>

  inline T              allsum

  ( Context&              ctx,
    const T&              val );




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   send
//-----------------------------------------------------------------------


template <class T>

  inline void           send

  ( Context&              ctx,
    const T&              val,
    int                   rank,
    int                   tag )

{
  ctx.send ( SendBuffer( &val, 1 ), rank, tag );
}


template <class T>

  inline void           send

  ( Context&              ctx,
    const T*              buf,
    idx_t                 len,
    int                   rank,
    int                   tag )

{
  ctx.send ( SendBuffer( buf, len ), rank, tag );
}


//-----------------------------------------------------------------------
//   recv
//-----------------------------------------------------------------------


template <class T>

  inline void           recv

  ( Context&              ctx,
    T&                    val,
    int                   rank,
    int                   tag,
    Status*               stat )

{
  ctx.recv ( RecvBuffer( &val, 1 ), rank, tag, stat );
}


template <class T>

  inline void           recv

  ( Context&              ctx,
    T*                    buf,
    idx_t                 len,
    int                   rank,
    int                   tag,
    Status*               stat )

{
  ctx.recv ( RecvBuffer( buf, len ), rank, tag, stat );
}


//-----------------------------------------------------------------------
//   allreduce
//-----------------------------------------------------------------------


template <class T>

  inline T              allreduce

  ( Context&              ctx,
    const T&              val,
    Opcode                op )

{
  T  ret;

  ctx.allreduce ( RecvBuffer( &ret, 1 ),
                  SendBuffer( &val, 1 ), op );

  return ret;
}


//-----------------------------------------------------------------------
//   allmax
//-----------------------------------------------------------------------


template <class T>

  inline T              allmax

  ( Context&              ctx,
    const T&              val )

{
  return allreduce ( ctx, val, MAX );
}


//-----------------------------------------------------------------------
//   allmin
//-----------------------------------------------------------------------


template <class T>

  inline T              allmin

  ( Context&              ctx,
    const T&              val )

{
  return allreduce ( ctx, val, MIN );
}


//-----------------------------------------------------------------------
//   allsum
//-----------------------------------------------------------------------


template <class T>

  inline T              allsum

  ( Context&              ctx,
    const T&              val )

{
  return allreduce ( ctx, val, SUM );
}


JEM_END_PACKAGE( mp )

#endif
