
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

#ifndef JEM_BASE_MEMORY_H
#define JEM_BASE_MEMORY_H

#include <new>
#include <cstring>
#include <jem/base/utilities.h>
#include <jem/base/TypeTraits.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   memory-related functions
//-----------------------------------------------------------------------


template <class T>

  inline void     construct

  ( T*              p,
    idx_t           n );

template <class T, class InputIterator>

  void            construct

  ( T*              p,
    InputIterator   first,
    InputIterator   last );

template <class T>

  inline void     construct

  ( T*              p,
    T*              first,
    T*              last );

template <class T>

  inline void     construct

  ( T*              p,
    const T*        first,
    const T*        last );

template <class T>

  inline void     destroy

  ( T*              p,
    idx_t           n );

template <class T, class InputIterator>

  void            copy

  ( T*              dest,
    InputIterator   first,
    InputIterator   last );

template <class T>

  inline void     copy

  ( T*              dest,
    T*              first,
    T*              last );

template <class T>

  inline void     copy

  ( T*              dest,
    const T*        first,
    const T*        last );





//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   private functions
//=======================================================================

//-----------------------------------------------------------------------
//   construct_
//-----------------------------------------------------------------------


template <class T>

  inline void     construct_

  ( T*              p,
    idx_t           n,
    const True&     fast ) noexcept

{}


template <class T>

  void            construct_

  ( T*              p,
    idx_t           n,
    const False&    fast )

{
  idx_t  i;

  try
  {
    for ( i = 0; i < n; i++ )
    {
      new ( &p[i] ) T();
    }
  }
  catch( ... )
  {
    jem::destroy ( p, i );
    throw;
  }
}


template <class T>

  inline void     construct_

  ( T*              p,
    const T*        first,
    const T*        last,
    const True&     fast ) noexcept

{
  std::memcpy ( p, first, sizeof(T) * (size_t) (last - first ) );
}


template <class T>

  void            construct_

  ( T*              p,
    const T*        first,
    const T*        last,
    const False&    fast )

{
  idx_t  i;

  try
  {
    for ( i = 0; first != last; first++, i++ )
    {
      new ( &p[i] ) T( *first );
    }
  }
  catch ( ... )
  {
    jem::destroy ( p, i );
    throw;
  }
}


//-----------------------------------------------------------------------
//   destroy_
//-----------------------------------------------------------------------


template <class T>

  inline void     destroy_

  ( T*              p,
    idx_t           n,
    const True&     fast ) noexcept

{}


template <class T>

  void            destroy_

  ( T*              p,
    idx_t           n,
    const False&    fast )

{
  while ( n )
  {
    n--;
    (p + n)->~T();
  }
}


//-----------------------------------------------------------------------
//   copy_
//-----------------------------------------------------------------------


template <class T>

  inline void     copy_

  ( T*              p,
    const T*        first,
    const T*        last,
    const True&     fast ) noexcept

{
  std::memcpy ( p, first, sizeof(T) * (size_t) (last - first) );
}


template <class T>

  void            copy_

  ( T*              p,
    const T*        first,
    const T*        last,
    const False&    fast )

{
  for ( ; first != last; ++first, ++p )
  {
    *p = *first;
  }
}


//=======================================================================
//   public functions
//=======================================================================

//-----------------------------------------------------------------------
//   construct
//-----------------------------------------------------------------------


template <class T>

  inline void     construct

  ( T*              p,
    idx_t           n )

{
  construct_ ( p, n, HasTrivialCtor<T>() || IsPOD<T>() );
}


template <class T, class InputIterator>

  void            construct

  ( T*              p,
    InputIterator   first,
    InputIterator   last )

{
  idx_t  i;

  try
  {
    for ( i = 0; first != last; ++first, ++i )
    {
      new ( &p[i] ) T( *first );
    }
  }
  catch ( ... )
  {
    destroy ( p, i );
    throw;
  }
}


template <class T>

  inline void     construct

  ( T*              p,
    T*              first,
    T*              last )

{
  construct_ ( p,
               const_cast<const T*> ( first ),
               const_cast<const T*> ( last  ),
               HasTrivialCopy<T>    () );
}


template <class T>

  inline void     construct

  ( T*              p,
    const T*        first,
    const T*        last )

{
  construct_ ( p, first, last, HasTrivialCopy<T>() || IsPOD<T>() );
}


//-----------------------------------------------------------------------
//   destroy
//-----------------------------------------------------------------------


template <class T>

  inline void     destroy

  ( T*              p,
    idx_t           n )

{
  destroy_ ( p, n, HasTrivialDtor<T>() || IsPOD<T>() );
}


//-----------------------------------------------------------------------
//   copy
//-----------------------------------------------------------------------


template <class T, class InputIterator>

  void            copy

  ( T*              p,
    InputIterator   first,
    InputIterator   last )

{
  for ( ; first != last; ++first, ++p )
  {
    *p = *first;
  }
}


template <class T>

  inline void     copy

  ( T*              p,
    T*              first,
    T*              last )

{
  copy_ ( p,
          const_cast <const T*> ( first ),
          const_cast <const T*> ( last  ),
          HasTrivialCopy<T>() || IsPOD<T>() );
}


template <class T>

  inline void     copy

  ( T*              p,
    const T*        first,
    const T*        last )

{
  copy_ ( p, first, last, HasTrivialCopy<T>() || IsPOD<T>() );
}


JEM_END_PACKAGE_BASE

#endif
