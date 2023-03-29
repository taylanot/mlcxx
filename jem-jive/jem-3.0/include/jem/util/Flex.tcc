
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

#ifndef JEM_UTIL_FLEX_TCC
#define JEM_UTIL_FLEX_TCC

#include <jem/base/assert.h>
#include <jem/base/limits.h>
#include <jem/base/memory.h>
#include <jem/base/utilities.h>
#include <jem/base/MemCache.h>
#include <jem/util/error.h>
#include <utility>

JEM_BEGIN_PACKAGE( util )


//=======================================================================
//   class Flex
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


template <class T>

  const float Flex<T>::EXPANSION_FACTOR = 1.5;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


template <class T>

  Flex<T>::Flex () noexcept

{
  data_     = nullptr;
  size_     = 0;
  capacity_ = 0;
  expand_   = EXPANSION_FACTOR;
}


template <class T>

  Flex<T>::Flex ( idx_t n )

{
  JEM_ASSERT2 ( n >= 0, "invalid Flex size" );

  data_     = (T*) MemCache::alloc ( (size_t) n * sizeof(T) );
  size_     = n;
  capacity_ = size_;
  expand_   = EXPANSION_FACTOR;

  try
  {
    construct ( data_, size_ );
  }
  catch ( ... )
  {
    MemCache::dealloc ( data_, (size_t) capacity_ * sizeof(T) );
    throw;
  }
}


template <class T>

  inline Flex<T>::Flex ( const Flex& rhs )

{
  init_ ( rhs.begin(), rhs.end() );

  expand_ = rhs.expand_;
}


template <class T>

  Flex<T>::Flex ( Flex&& rhs ) noexcept

{
  data_         = rhs.data_;
  size_         = rhs.size_;
  capacity_     = rhs.capacity_;
  expand_       = rhs.expand_;

  rhs.data_     = nullptr;
  rhs.size_     = 0;
  rhs.capacity_ = 0;
}


template <class T>
  template <class InputIterator>

  inline Flex<T>::Flex

  ( InputIterator  first,
    InputIterator  last )

{
  init_ ( first, last );

  expand_ = EXPANSION_FACTOR;
}

template <class T>
  template <class U>

  inline Flex<T>::Flex ( std::initializer_list<U> list )

{
  init_ ( list.begin(), list.end() );

  expand_ = EXPANSION_FACTOR;
}


template <class T>

  Flex<T>::~Flex ()

{
  if ( data_ )
  {
    destroy           ( data_, size_ );
    MemCache::dealloc ( data_, (size_t) capacity_ * sizeof(T) );
  }

  data_ = nullptr;
}


//-----------------------------------------------------------------------
//   begin & end
//-----------------------------------------------------------------------


template <class T>

  inline typename
  Flex<T>::Iterator Flex<T>::begin () noexcept

{
  return data_;
}


template <class T>

  inline typename
  Flex<T>::ConstIterator Flex<T>::begin () const noexcept

{
  return data_;
}


template <class T>

  inline typename
  Flex<T>::Iterator Flex<T>::end () noexcept

{
  return (data_ + size_);
}


template <class T>

  inline typename
  Flex<T>::ConstIterator Flex<T>::end () const noexcept

{
  return (data_ + size_);
}


//-----------------------------------------------------------------------
//   front & back
//-----------------------------------------------------------------------


template <class T>

  inline T& Flex<T>::front ()

{
  JEM_ASSERT2 ( size() > 0, "empty Flex"  );

  return *data_;
}


template <class T>

  inline const T& Flex<T>::front () const

{
  JEM_ASSERT2 ( size() > 0, "empty Flex" );

  return *data_;
}


template <class T>

  inline T& Flex<T>::back ()

{
  JEM_ASSERT2 ( size() > 0, "empty Flex" );

  return data_[size_ - 1];
}


template <class T>

  inline const T& Flex<T>::back () const

{
  JEM_ASSERT2 ( size() > 0, "empty Flex" );

  return data_[size_ - 1];
}


//-----------------------------------------------------------------------
//   assignment operators
//-----------------------------------------------------------------------


template <class T>

  Flex<T>& Flex<T>::operator = ( const Flex& rhs )

{
  if ( this != &rhs )
  {
    if ( size_ != rhs.size() )
    {
      resize ( rhs.size_ );
    }

    copy ( data_, rhs.begin(), rhs.end() );

    expand_ = rhs.expand_;
  }

  return *this;
}


template <class T>

  inline Flex<T>& Flex<T>::operator = ( Flex&& rhs ) noexcept

{
  swap ( rhs );

  return *this;
}


template <class T>

  Flex<T>& Flex<T>::operator = ( T rhs )

{
  const idx_t  n    = size_;
  T*           dest = data_;;

  for ( idx_t i = 0; i < n; i++ )
  {
    dest[i] = rhs;
  }

  return *this;
}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


template <class T>

  void Flex<T>::swap ( Flex& rhs ) noexcept

{
  jem::swap (     data_,     rhs.data_ );
  jem::swap (     size_,     rhs.size_ );
  jem::swap ( capacity_, rhs.capacity_ );
  jem::swap (   expand_,   rhs.expand_ );
}


//-----------------------------------------------------------------------
//   subscript operators
//-----------------------------------------------------------------------


template <class T>

  inline T& Flex<T>::operator [] ( idx_t i )

{
  JEM_ASSERT2 ( checkIndex( i, size() ), "invalid Flex index" );

  return data_[i];
}


template <class T>

  inline const T& Flex<T>::operator [] ( idx_t i ) const

{
  JEM_ASSERT2 ( checkIndex( i, size() ), "invalid Flex index" );

  return data_[i];
}


//-----------------------------------------------------------------------
//   extend
//-----------------------------------------------------------------------


template <class T>

  T* Flex<T>::extend ( idx_t n )

{
  JEM_ASSERT2 ( n >= 0, "invalid extension size" );

  if ( size_ + n > capacity_ )
  {
    grow_ ( size_ + n );
  }

  T*  p = data_ + size_;

  jem::construct ( p, n );

  size_ += n;

  return p;
}


//-----------------------------------------------------------------------
//   pushBack
//-----------------------------------------------------------------------


template <class T>

  inline void Flex<T>::pushBack ( const T& item )

{
  if ( size_ == capacity_ )
  {
    grow_ ( size_ + 1 );
  }

  new ( &data_[size_] ) T( item );

  size_++;
}


template <class T>

  inline void Flex<T>::pushBack ( T&& item )

{
  if ( size_ == capacity_ )
  {
    grow_ ( size_ + 1 );
  }

  new ( &data_[size_] ) T ( std::move( item ) );

  size_++;
}


template <class T>

  void Flex<T>::pushBack ( const T& item, idx_t n )

{
  JEM_ASSERT2 ( n >= 0, "invalid item repeat count" );

  if ( size_ + n > capacity_ )
  {
    grow_ ( size_ + n );
  }

  for ( idx_t i = 0; i < n; i++ )
  {
    new ( &data_[size_] ) T( item );

    size_++;
  }
}


template <class T>
  template <class InputIterator>

  void Flex<T>::pushBack

  ( InputIterator  first,
    InputIterator  last )

{
  const idx_t  n = distance ( first, last );

  if ( size_ + n > capacity_ )
  {
    grow_ ( size_ + n );
  }

  construct ( data_ + size_, first, last );

  size_ += n;
}


template <class T>
  template <class U>

  inline void Flex<T>::pushBack

  ( std::initializer_list<U>  list )

{
  pushBack ( list.begin(), list.end() );
}


//-----------------------------------------------------------------------
//   popBack
//-----------------------------------------------------------------------


template <class T>

  inline void Flex<T>::popBack ()

{
  JEM_ASSERT2 ( size() > 0, "empty Flex" );

  size_--;

  (data_ + size_)->~T();
}


template <class T>

  inline void Flex<T>::popBack ( idx_t n )

{
  JEM_ASSERT2 ( n >= 0, "invalid number of items to be removed" );

  size_ -= n;

  destroy ( data_ + size_, n );
}


//-----------------------------------------------------------------------
//   insert
//-----------------------------------------------------------------------


template <class T>

  inline void Flex<T>::insert ( Iterator pos, const T& item )

{
  JEM_ASSERT2 ( pos >= begin() && pos <= end(),
                "invalid Flex iterator" );

  idx_t  offset = (idx_t) (pos - data_);

  shiftRight_ ( offset, 1 );

  data_[offset] = item;
}


template <class T>

  inline void Flex<T>::insert ( Iterator pos, T&& item )

{
  JEM_ASSERT2 ( pos >= begin() && pos <= end(),
                "invalid Flex iterator" );

  idx_t  offset = (idx_t) (pos - data_);

  shiftRight_ ( offset, 1 );

  data_[offset] = std::move ( item );
}


template <class T>

  void Flex<T>::insert ( Iterator pos, T item, idx_t n )

{
  JEM_ASSERT2 ( pos >= begin() && pos <= end(),
                "invalid Flex iterator" );

  idx_t  offset = (idx_t) (pos - data_);
  T*    dest;


  shiftRight_ ( offset, n );

  dest = data_ + offset;

  for ( idx_t i = 0; i < n; i++ )
  {
    dest[i] = item;
  }
}


template <class T>
  template <class InputIterator>

  void Flex<T>::insert

  ( Iterator       pos,
    InputIterator  first,
    InputIterator  last )

{
  JEM_ASSERT2 ( pos >= begin() && pos <= end(),
                "invalid Flex iterator" );

  idx_t  offset = (idx_t) (pos - data_);

  shiftRight_ ( offset, (idx_t) distance( first, last ) );

  copy ( data_ + offset, first, last );
}


template <class T>
  template <class U>

  inline void Flex<T>::insert

  ( Iterator                  pos,
    std::initializer_list<U>  list )

{
  insert ( pos, list.begin(), list.end() );
}


//-----------------------------------------------------------------------
//   erase
//-----------------------------------------------------------------------


template <class T>

  inline void Flex<T>::erase ( Iterator pos )

{
  JEM_ASSERT2 ( pos >= begin() && pos <= end(),
                "invalid Flex iterator" );

  shiftLeft_ ( (idx_t) (pos - data_), 1 );
}


template <class T>

  inline void Flex<T>::erase ( Iterator first, Iterator last )

{
  JEM_ASSERT2 ( first >= begin() && first <= end(),
                "invalid Flex iterator" );

  shiftLeft_ ( (idx_t) (first - data_),
               (idx_t) (last  - first) );
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


template <class T>

  inline void Flex<T>::clear ()

{
  destroy ( data_, size_ );

  size_ = 0;
}


//-----------------------------------------------------------------------
//   resize
//-----------------------------------------------------------------------


template <class T>

  void Flex<T>::resize ( idx_t n )

{
  if ( n > size_ )
  {
    if ( n > capacity_ )
    {
      setCapacity_ ( n );
    }

    construct ( data_ + size_, n - size_ );

    size_ = n;
  }
  else if ( n < size_ )
  {
    popBack ( size_ - n );
  }
}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


template <class T>

  inline void Flex<T>::reserve ( idx_t cap )

{
  if ( cap > capacity_ )
  {
    setCapacity_ ( cap );
  }
}


//-----------------------------------------------------------------------
//   setExpansionFactor
//-----------------------------------------------------------------------


template <class T>

  inline void Flex<T>::setExpansionFactor ( float x )

{
  JEM_ASSERT2 ( x > 0.0, "invalid Flex expansion factor" );

  expand_ = x;
}


//-----------------------------------------------------------------------
//   getExpansionFactor
//-----------------------------------------------------------------------


template <class T>

  inline float Flex<T>::getExpansionFactor () const noexcept

{
  return expand_;
}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


template <class T>

  inline void Flex<T>::trimToSize ()

{
  if ( capacity_ > size_ )
  {
    setCapacity_ ( size_ );
  }
}


//-----------------------------------------------------------------------
//   capacity
//-----------------------------------------------------------------------


template <class T>

  inline idx_t Flex<T>::capacity () const noexcept

{
  return capacity_;
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


template <class T>

  inline idx_t Flex<T>::size () const noexcept

{
  return size_;
}


//-----------------------------------------------------------------------
//   addr
//-----------------------------------------------------------------------


template <class T>

  inline T* Flex<T>::addr ( idx_t i ) noexcept

{
  return (data_ + i);
}


template <class T>

  inline const T* Flex<T>::addr ( idx_t i ) const noexcept

{
  return (data_ + i);
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


template <class T>
  template <class InputIterator>

  void Flex<T>::init_

  ( InputIterator  first,
    InputIterator  last )

{
  size_ = capacity_ = distance ( first, last );
  data_ = (T*) MemCache::alloc ( (size_t) size_ * sizeof(*data_) );

  try
  {
    construct ( data_, first, last );
  }
  catch ( ... )
  {
    MemCache::dealloc ( data_,
                        (size_t) capacity_ * sizeof(*data_) );
    throw;
  }
}


//-----------------------------------------------------------------------
//   shiftRight_
//-----------------------------------------------------------------------


template <class T>

  void Flex<T>::shiftRight_ ( idx_t pos, idx_t n )

{
  idx_t i;

  if ( n <= 0 )
  {
    return;
  }

  if ( size_ + n > capacity_ )
  {
    grow_ ( size_ + n );
  }

  construct ( data_ + size_, n );

  i      = size_ - 1;
  size_ += n;

  for ( ; i >= pos; i-- )
  {
    data_[i + n] = data_[i];
  }
}


//-----------------------------------------------------------------------
//   shiftLeft_
//-----------------------------------------------------------------------


template <class T>

  void Flex<T>::shiftLeft_ ( idx_t pos, idx_t n )

{
  idx_t  k = size_ - n;
  idx_t  i;

  if ( n <= 0 )
  {
    return;
  }

  for ( i = pos; i < k; i++ )
  {
    data_[i] = data_[i + n];
  }

  destroy ( data_ + k, n );

  size_ = k;
}


//-----------------------------------------------------------------------
//   grow_
//-----------------------------------------------------------------------


template <class T>

  void Flex<T>::grow_ ( idx_t cap )

{
  idx_t  n = (idx_t) ((double) maxOf( capacity_ ) / (expand_ + 0.01));

  if ( capacity_ >= n )
  {
    n = maxOf ( capacity_ );
  }
  else
  {
    idx_t  k = (idx_t) ((double) expand_ * (double) capacity_);

    n = jem::max ( k, cap );
  }

  if ( cap < 0 || n < cap )
  {
    overflowError ( JEM_FUNC, capacity_ );
  }

  setCapacity_ ( n );
}


//-----------------------------------------------------------------------
//   setCapacity_
//-----------------------------------------------------------------------


template <class T>

  inline void Flex<T>::setCapacity_ ( idx_t cap )

{
  setCapacity_ (
    cap,

    // An explicit operator function call is required because GCC may
    // get confused if the && operator is applied directly.

    operator && ( HasTrivialDtor<T>(), HasTrivialCopy<T>() ) ||
    IsPOD<T>()
  );
}


//-----------------------------------------------------------------------
//   setCapacity_ (using realloc)
//-----------------------------------------------------------------------


template <class T>

  inline void Flex<T>::setCapacity_

  ( idx_t         cap,
    const True&  realloc )

{
  data_ = (T*) MemCache::realloc

    ( data_,
      (size_t) capacity_ * sizeof(*data_),
      (size_t) cap       * sizeof(*data_) );

  capacity_ = cap;
}


//-----------------------------------------------------------------------
//   setCapacity_ (without using realloc)
//-----------------------------------------------------------------------


template <class T>

  void Flex<T>::setCapacity_

  ( idx_t          cap,
    const False&  realloc )

{
  T*  buffer = (T*)

    MemCache::alloc ( (size_t) cap * sizeof(*buffer) );

  try
  {
    construct ( buffer, data_, data_ + size_ );
  }
  catch ( ... )
  {
    MemCache::dealloc ( buffer, (size_t) cap * sizeof(*buffer) );
    throw;
  }

  jem::swap (     data_, buffer );
  jem::swap ( capacity_,    cap );

  destroy           ( buffer, size_ );
  MemCache::dealloc ( buffer, (size_t) cap * sizeof(*buffer) );
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   decode
//-----------------------------------------------------------------------


template <class Input, class T>

  void              decode

  ( Input&            in,
    Flex<T>&          flx )

{
  float   x;
  uidx_t  sz;
  idx_t   n;

  decode ( in, sz, x );

  n = (idx_t) sz;

  flx.setExpansionFactor ( x );
  flx.resize             ( n );

  decodeArray ( in, flx.addr(), n );
}


//-----------------------------------------------------------------------
//   encode
//-----------------------------------------------------------------------


template <class Output, class T>

  void              encode

  ( Output&           out,
    const Flex<T>&    flx )

{
  encode      ( out, flx.size(), flx.getExpansionFactor() );
  encodeArray ( out, flx.addr(), flx.size() );
}


//-----------------------------------------------------------------------
//   print
//-----------------------------------------------------------------------


template <class Output, class T>

  void              print

  ( Output&           out,
    const Flex<T>&    flx )

{
  if ( flx.size() > 0 )
  {
    print         ( out, "[ " );
    printSequence ( out, flx.begin(), flx.end(), ", " );
    print         ( out, " ]" );
  }
  else
  {
    print         ( out, "[]" );
  }
}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


template <class T>

  inline void   swap

  ( Flex<T>&      lhs,
    Flex<T>&      rhs ) noexcept

{
  lhs.swap ( rhs );
}


JEM_END_PACKAGE( util )

JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   class TypeTraits< util::Flex<T> >
//=======================================================================


template <class T>

  class TypeTraits< util::Flex<T> > : public BasicTypeTraits

{
 public:

  static const bool   IS_SERIALIZABLE = IsSerializable<T>::VALUE;

  static String       whatis   ()
  {
    return TypeTraits<T>::whatis() + String ( " flex" );
  }

  static String       typeName ()
  {
    return ( String( "jem::util::Flex<" ) +
             TypeTraits<T>::typeName ()   +
             String( '>' ) );
  }

};


JEM_END_PACKAGE_BASE

#endif
