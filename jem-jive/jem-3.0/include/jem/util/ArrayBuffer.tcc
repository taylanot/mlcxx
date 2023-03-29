
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

#ifndef JEM_UTIL_ARRAYBUFFER_TCC
#define JEM_UTIL_ARRAYBUFFER_TCC

#include <utility>
#include <jem/base/limits.h>
#include <jem/util/error.h>


JEM_BEGIN_PACKAGE( util )


//=======================================================================
//   class ArrayBuffer
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


template <class T>

  const float ArrayBuffer<T>::EXPANSION_FACTOR = 1.5;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


template <class T>

  ArrayBuffer<T>::ArrayBuffer () noexcept

{
  init_ ( 0 );
}


template <class T>

  ArrayBuffer<T>::ArrayBuffer ( idx_t n ) :

    buffer_ ( n )

{
  init_ ( n );
}


template <class T>

  ArrayBuffer<T>::ArrayBuffer ( const Array<T>& rhs ) :

    buffer_ ( rhs )

{
  init_ ( rhs.size() );
}


template <class T>

  ArrayBuffer<T>::ArrayBuffer ( const Array<T>& buf, idx_t n ) :

    buffer_ ( buf )

{
  JEM_ASSERT2 ( n <= buf.size(),
                "ArrayBuffer size larger than Array size" );

  init_ ( n );
}


template <class T>

  ArrayBuffer<T>::ArrayBuffer ( const ArrayBuffer& rhs ) :

    buffer_ ( rhs.buffer_.clone() )

{
  size_   = rhs.size_;
  expand_ = rhs.expand_;
}


template <class T>

  ArrayBuffer<T>::ArrayBuffer ( ArrayBuffer&& rhs ) noexcept :

    buffer_ ( std::move( rhs.buffer_ ) )

{
  size_   = rhs.size_;
  expand_ = rhs.expand_;
}


template <class T>
  template <class InputIterator>

  ArrayBuffer<T>::ArrayBuffer

  ( InputIterator  first,
    InputIterator  last ) :

    buffer_ ( first, last )

{
  init_ ( buffer_.size() );
}


template <class T>
  template <class U>

  ArrayBuffer<T>::ArrayBuffer ( std::initializer_list<U> list ) :

    buffer_ ( list )

{
  init_ ( buffer_.size() );
}


//-----------------------------------------------------------------------
//   begin & end
//-----------------------------------------------------------------------


template <class T>

  inline typename ArrayBuffer<T>::Iterator
  ArrayBuffer<T>::begin () const noexcept

{
  return buffer_.begin ();
}


template <class T>

  inline typename ArrayBuffer<T>::Iterator
  ArrayBuffer<T>::end () const noexcept

{
  return ( buffer_.begin() + size_ );
}


//-----------------------------------------------------------------------
//   front & back
//-----------------------------------------------------------------------


template <class T>

  inline T& ArrayBuffer<T>::front () const

{
  JEM_ASSERT2 ( size() > 0, "empty ArrayBuffer" );

  return buffer_[0];
}


template <class T>

  inline T& ArrayBuffer<T>::back () const

{
  JEM_ASSERT2 ( size() > 0, "empty ArrayBuffer" );

  return buffer_[size_ - 1];
}


//-----------------------------------------------------------------------
//   assignment operators
//-----------------------------------------------------------------------


template <class T>

  ArrayBuffer<T>&
  ArrayBuffer<T>::operator = ( const ArrayBuffer& rhs )

{
  if ( this != &rhs )
  {
    if ( size_ != rhs.size_ )
    {
      resize ( rhs.size_ );
    }

    buffer_[slice(BEGIN,size_)] = rhs.buffer_ [slice(BEGIN,size_)];

    expand_ = rhs.expand_;
  }

  return *this;
}


template <class T>

  inline ArrayBuffer<T>&
  ArrayBuffer<T>::operator = ( ArrayBuffer&& rhs ) noexcept

{
  swap ( rhs );

  return *this;
}


template <class T>

  inline const ArrayBuffer<T>&
  ArrayBuffer<T>::operator = ( T rhs ) const

{
  buffer_[slice(BEGIN,size_)] = rhs;

  return *this;
}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


template <class T>

  void ArrayBuffer<T>::swap ( ArrayBuffer& rhs ) noexcept

{
  buffer_.swap ( rhs.buffer_ );

  jem::swap (   size_,   rhs.size_ );
  jem::swap ( expand_, rhs.expand_ );
}


//-----------------------------------------------------------------------
//   subscript operator
//-----------------------------------------------------------------------


template <class T>
  template <class I>

  inline typename
  array::SliceTraits<T,I>::ResultType
  ArrayBuffer<T>::operator [] ( const I& i ) const

{
  return buffer_[i];
}


//-----------------------------------------------------------------------
//   pushBack
//-----------------------------------------------------------------------


template <class T>

  inline void ArrayBuffer<T>::pushBack ( const T& item )

{
  if ( size_ == buffer_.size() )
  {
    grow_ ( size_ + 1 );
  }

  buffer_[size_] = item;
  size_++;
}


template <class T>

  inline void ArrayBuffer<T>::pushBack ( T&& item )

{
  if ( size_ == buffer_.size() )
  {
    grow_ ( size_ + 1 );
  }

  buffer_[size_] = std::move ( item );
  size_++;
}


template <class T>

  void ArrayBuffer<T>::pushBack ( T item, idx_t n )

{
  JEM_ASSERT2 ( n >= 0, "invalid item repeat count" );

  if ( size_ + n > buffer_.size() )
  {
    grow_( size_ + n );
  }

  buffer_[slice(size_,size_ + n)] = item;
  size_ += n;
}


template <class T>
  template <class InputIterator>

  void ArrayBuffer<T>::pushBack

  ( InputIterator  first,
    InputIterator  last )

{
  const idx_t  n = distance ( first, last );

  if ( size_ + n > buffer_.size() )
  {
    grow_( size_ + n );
  }

  if ( buffer_.isContiguous() )
  {
    copy ( buffer_.addr( size_ ), first, last );
    size_ += n;
  }
  else
  {
    for ( ; first != last; ++first, ++size_ )
    {
      buffer_[size_] = *first;
    }
  }
}


template <class T>
  template <class U>

  inline void
  ArrayBuffer<T>::pushBack ( std::initializer_list<U> list )

{
  pushBack ( list.begin(), list.end() );
}


//-----------------------------------------------------------------------
//   popBack
//-----------------------------------------------------------------------


template <class T>

  inline void ArrayBuffer<T>::popBack ()

{
  JEM_ASSERT2 ( size() > 0, "empty ArrayBuffer" );

  size_--;
}


template <class T>

  inline void ArrayBuffer<T>::popBack ( idx_t n )

{
  JEM_ASSERT2 ( n >= 0 && n <= size(),
                "invalid number of items to be removed" );

  size_ -= n;
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


template <class T>

  inline void ArrayBuffer<T>::clear ()

{
  size_ = 0;
}


//-----------------------------------------------------------------------
//   resize
//-----------------------------------------------------------------------


template <class T>

  inline void ArrayBuffer<T>::resize ( idx_t n )

{
  if ( n > buffer_.size() )
  {
    buffer_.reshape ( n );
  }

  size_ = n;
}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


template <class T>

  inline void ArrayBuffer<T>::reserve ( idx_t cap )

{
  if ( cap > buffer_.size() )
  {
    buffer_.reshape ( cap );
  }
}


//-----------------------------------------------------------------------
//   setExpansionFactor
//-----------------------------------------------------------------------


template <class T>

  inline void ArrayBuffer<T>::setExpansionFactor ( float x )

{
  expand_ = x;
}


//-----------------------------------------------------------------------
//   getExpansionFactor
//-----------------------------------------------------------------------


template <class T>

  inline float ArrayBuffer<T>::getExpansionFactor () const noexcept

{
  return expand_;
}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


template <class T>

  inline void ArrayBuffer<T>::trimToSize ()

{
  if ( buffer_.size() > size_ )
  {
    buffer_.reshape ( size_ );
  }
}


//-----------------------------------------------------------------------
//   capacity
//-----------------------------------------------------------------------


template <class T>

  inline idx_t ArrayBuffer<T>::capacity () const noexcept

{
  return buffer_.size ();
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


template <class T>

  inline idx_t ArrayBuffer<T>::size () const noexcept

{
  return size_;
}


//-----------------------------------------------------------------------
//   toArray
//-----------------------------------------------------------------------


template <class T>

  inline Array<T> ArrayBuffer<T>::toArray () const

{
  return buffer_[slice( BEGIN,size_)];
}


//-----------------------------------------------------------------------
//   getBuffer
//-----------------------------------------------------------------------


template <class T>

  inline const Array<T>& ArrayBuffer<T>::getBuffer () const noexcept

{
  return buffer_;
}


//-----------------------------------------------------------------------
//   addr
//-----------------------------------------------------------------------


template <class T>

  inline T* ArrayBuffer<T>::addr () const noexcept

{
  return buffer_.addr ();
}


template <class T>

  inline T* ArrayBuffer<T>::addr ( idx_t offset ) const

{
  return buffer_.addr ( offset );
}


//-----------------------------------------------------------------------
//   grow_
//-----------------------------------------------------------------------


template <class T>

  void ArrayBuffer<T>::grow_ ( idx_t cap )

{
  idx_t  k = buffer_.size ();
  idx_t  n = (idx_t) ((double) maxOf( k ) / (expand_ + 0.01));

  if ( k >= n )
  {
    n = maxOf ( k );
  }
  else
  {
    n = jem::max( (idx_t) ( expand_ * (double) k ), cap );
  }

  if ( cap < 0 || n < cap )
  {
    overflowError ( JEM_FUNC, k );
  }

  buffer_.reshape ( n );
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


template <class T>

  inline void ArrayBuffer<T>::init_ ( idx_t n )

{
  size_   = n;
  expand_ = EXPANSION_FACTOR;
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   decode
//-----------------------------------------------------------------------


template <class Input, class T>

  void                    decode

  ( Input&                  in,
    ArrayBuffer<T>&         buf )

{
  Array<T>  a;
  float     x;
  uidx_t    sz;

  decode ( in, sz, x );

  idx_t  n = (idx_t) sz;

  buf.setExpansionFactor ( x );
  buf.resize             ( n );
  a.ref                  ( buf.toArray() );

  if ( a.isContiguous() )
  {
    decodeArray ( in, a.addr(), n );
  }
  else
  {
    for ( idx_t i = 0; i < n; i++ )
    {
      decode    ( in, a[i] );
    }
  }
}


//-----------------------------------------------------------------------
//   encode
//-----------------------------------------------------------------------


template <class Output, class T>

  void                    encode

  ( Output&                 out,
    const ArrayBuffer<T>&   buf )

{
  Array<T>  a ( buf.toArray() );

  encode ( out, buf.size(), buf.getExpansionFactor() );

  if ( a.isContiguous() )
  {
    encodeArray ( out, a.addr(), a.size() );
  }
  else
  {
    for ( idx_t i = 0; i < a.size(); i++ )
    {
      encode    ( out, a[i] );
    }
  }
}


//-----------------------------------------------------------------------
//   print
//-----------------------------------------------------------------------


template <class Output, class T>

  void                    print

  ( Output&                 out,
    const ArrayBuffer<T>&   buf )

{
  if ( buf.size() > 0 )
  {
    print         ( out, "[ " );
    printSequence ( out, buf.begin(), buf.end(), ", " );
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

  inline void       swap

  ( ArrayBuffer<T>&   lhs,
    ArrayBuffer<T>&   rhs ) noexcept

{
  lhs.swap ( rhs );
}


JEM_END_PACKAGE( util )

JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   class TypeTraits< util::ArrayBuffer<T> >
//=======================================================================


template <class T>

  class TypeTraits< util::ArrayBuffer<T> > : public BasicTypeTraits

{
 public:

  static const bool   IS_SERIALIZABLE = IsSerializable<T>::VALUE;

  static String       whatis   ()
  {
    return TypeTraits<T>::whatis() + String ( " array buffer" );
  }

  static String       typeName ()
  {
    return ( String ( "jem::util::ArrayBuffer<" ) +
             TypeTraits<T>::typeName ()           +
             String ( '>' ) );
  }

};


JEM_END_PACKAGE_BASE

#endif
