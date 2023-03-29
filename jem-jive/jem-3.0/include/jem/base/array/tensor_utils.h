
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

#ifndef JEM_BASE_ARRAY_TENSOR_UTILS_H
#define JEM_BASE_ARRAY_TENSOR_UTILS_H

#include <jem/base/tuple/Tuple.h>
#include <jem/base/array/base.h>


JEM_BEGIN_PACKAGE_BASE
JEM_BEGIN_NAMESPACE( tensor )


//-----------------------------------------------------------------------
//   error functions
//-----------------------------------------------------------------------


void                      uboundsError

  ( idx_t                   ub1,
    idx_t                   ub2 );

void                      unboundedError  ();
void                      usedIndexError  ();


//-----------------------------------------------------------------------
//   checkUbounds
//-----------------------------------------------------------------------


inline void               checkUbounds

  ( idx_t                    ub1,
    idx_t                    ub2 )

{
  if ( ub1 != ub2 && ub1 >= 0_idx && ub2 >= 0_idx )
  {
    uboundsError ( ub1, ub2 );
  }
}


//-----------------------------------------------------------------------
//   select
//-----------------------------------------------------------------------


template <class T, int N>

  inline T                select

  ( const Tuple<idx_t*,N>&  index,
    idx_t*                  iaxis,
    const Tuple<T,N>&       values,
    T                       altval )

{
  for ( int i = 0; i < N; i++ )
  {
    if ( index[i] == iaxis )
    {
      return values[i];
    }
  }

  return altval;
}


template <class T>

  inline T                select

  ( const Tuple<idx_t*,1>&  index,
    idx_t*                  iaxis,
    const Tuple<T,1>&       values,
    T                       altval )

{
  return ( (index[0] == iaxis) ? values[0] : altval );
}


template <class T>

  inline T                select

  ( const Tuple<idx_t*,2>&  index,
    idx_t*                  iaxis,
    const Tuple<T,2>&       values,
    T                       altval )

{
  if      ( index[0] == iaxis )
  {
    return values[0];
  }
  else if ( index[1] == iaxis )
  {
    return values[1];
  }
  else
  {
    return altval;
  }
}



//-----------------------------------------------------------------------
//   ioffset
//-----------------------------------------------------------------------


template <int N>

  inline idx_t            ioffset

  ( const Tuple<idx_t*,N>&  index,
    const Tuple<idx_t,N>&   stride )

{
  idx_t  k = 0;

  for ( int i = 0; i < N; i++ )
  {
    k += (*index[i]) * stride[i];
  }

  return k;
}


inline idx_t              ioffset

  ( const Tuple<idx_t*,1>&  index,
    const Tuple<idx_t,1>&   stride )

{
  return ( (*index[0]) * stride[0] );
}


inline idx_t              ioffset

  ( const Tuple<idx_t*,2>&  index,
    const Tuple<idx_t,2>&   stride )

{
  return ( (*index[0]) * stride[0] +
           (*index[1]) * stride[1] );
}


inline idx_t              ioffset

  ( const Tuple<idx_t*,3>&  index,
    const Tuple<idx_t,3>&   stride )

{
  return ( (*index[0]) * stride[0] +
           (*index[1]) * stride[1] +
           (*index[2]) * stride[2] );
}


inline idx_t              ioffset

  ( const Tuple<idx_t*,4>&  index,
    const Tuple<idx_t,4>&   stride )

{
  return ( (*index[0]) * stride[0] +
           (*index[1]) * stride[1] +
           (*index[2]) * stride[2] +
           (*index[3]) * stride[3] );
}


//-----------------------------------------------------------------------
//   incrIndex
//-----------------------------------------------------------------------


template <int N>

  inline bool             incrIndex

  ( Tuple<idx_t,N>&         index,
    const Tuple<idx_t,N>&   shape )

{
  return array::incrIndex ( index, shape );
}


inline bool               incrIndex

  ( Tuple<idx_t,1>&         index,
    const Tuple<idx_t,1>&   shape )

{
  return false;
}


inline bool               incrIndex

  ( Tuple<idx_t,2>&         index,
    const Tuple<idx_t,2>&   shape )

{
  return false;
}


bool                      incrIndex

  ( idx_t* const*           index,
    const idx_t*            ubound,
    int                     rank );


template <int N>

  inline bool             incrIndex

  ( const Tuple<idx_t*,N>&  index,
    const Tuple<idx_t,N>&   ubound )

{
  static_assert ( N > 3, "invalid dimension" );

  (*index[0]) = 0_idx;
  (*index[1]) = 0_idx;

  if ( (++(*index[2])) >= ubound[2] )
  {
    (*index[2]) = 0;

    if ( (++(*index[3])) >= ubound[3] )
    {
      return incrIndex ( index .addr() + 3,
                         ubound.addr() + 3, N - 3 );
    }
  }

  return true;
}


inline bool               incrIndex

  ( const Tuple<idx_t*,1>&  index,
    const Tuple<idx_t,1>&   ubound )

{
  return false;
}


inline bool               incrIndex

  ( const Tuple<idx_t*,2>&  index,
    const Tuple<idx_t,2>&   ubound )

{
  return false;
}


inline bool               incrIndex

  ( const Tuple<idx_t*,3>&  index,
    const Tuple<idx_t,3>&   ubound )

{
  (*index[0]) = 0_idx;
  (*index[1]) = 0_idx;
  (*index[2])++;

  return ( (*index[2]) < ubound[2] );
}


inline bool               incrIndex

  ( const Tuple<idx_t*,4>&  index,
    const Tuple<idx_t,4>&   ubound )

{
  (*index[0]) = 0_idx;
  (*index[1]) = 0_idx;

  if ( (++(*index[2])) >= ubound[2] )
  {
    (*index[2]) = 0_idx;
    (*index[3])++;

    return ( (*index[3]) < ubound[3] );
  }

  return true;
}


//-----------------------------------------------------------------------
//   zeroIndex
//-----------------------------------------------------------------------


void                      zeroIndex

  ( idx_t* const*           index,
    int                     rank );


template <int N>

  inline void             zeroIndex

  ( const Tuple<idx_t*,N>&  index )

{
  zeroIndex ( index.addr(), N );
}


inline void               zeroIndex

  ( const Tuple<idx_t*,1>&  index )

{
  (*index[0]) = 0_idx;
}

inline void               zeroIndex

  ( const Tuple<idx_t*,2>&  index )

{
  (*index[0]) = 0_idx;
  (*index[1]) = 0_idx;
}


inline void               zeroIndex

  ( const Tuple<idx_t*,3>&  index )

{
  (*index[0]) = 0_idx;
  (*index[1]) = 0_idx;
  (*index[2]) = 0_idx;
}


inline void               zeroIndex

  ( const Tuple<idx_t*,4>&  index )

{
  (*index[0]) = 0_idx;
  (*index[1]) = 0_idx;
  (*index[2]) = 0_idx;
  (*index[3]) = 0_idx;
}


//-----------------------------------------------------------------------
//   checkUnique
//-----------------------------------------------------------------------


void                      checkUnique

  ( idx_t* const*           index,
    int                     rank );


template <int N>

  inline void             checkUnique

  ( const Tuple<idx_t*,N>&  index )

{
  checkUnique ( index.addr(), N );
}


inline void               checkUnique

  ( const Tuple<idx_t*,1>&  index )

{}


inline void               checkUnique

  ( const Tuple<idx_t*,2>&  index )

{
  if ( index[0] == index[1] )
  {
    usedIndexError ();
  }
}


inline void               checkUnique

  ( const Tuple<idx_t*,3>&  index )

{
  if ( index[0] == index[1] ||
       index[0] == index[2] ||
       index[1] == index[2] )
  {
    usedIndexError ();
  }
}


inline void               checkUnique

  ( const Tuple<idx_t*,4>&  index )

{
  if ( index[0] == index[1] ||
       index[0] == index[2] ||
       index[0] == index[3] ||
       index[1] == index[2] ||
       index[1] == index[3] ||
       index[2] == index[3] )
  {
    usedIndexError ();
  }
}


//-----------------------------------------------------------------------
//   class Is2D
//-----------------------------------------------------------------------


template <int N>

  class Is1D : public BoolConstant<(N == 1)> {};


JEM_END_NAMESPACE( tensor )
JEM_END_PACKAGE_BASE

#endif
