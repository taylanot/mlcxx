
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

#ifndef JEM_BASE_ARRAY_ITERATOR_TCC
#define JEM_BASE_ARRAY_ITERATOR_TCC


JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   class ArrayIterator
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


template <class T, int N>

  inline ArrayIterator<T,N>::ArrayIterator () noexcept :

    data_   ( nullptr ),
    offset_ ( -1_idx )

{}


template <class T, int N>

  ArrayIterator<T,N>::ArrayIterator

  ( T*            data,
    const Shape&  shape,
    const Shape&  stride ) noexcept :

    data_   (   data ),
    offset_ (      0 ),
    shape_  (  shape ),
    stride_ ( stride )

{
  index_ = 0_idx;
}


template <class T, int N>

  ArrayIterator<T,N>::ArrayIterator

  ( T*            data,
    const Shape&  shape,
    const Shape&  stride,
    EndTag                 ) noexcept :

    data_   (   data ),
    offset_ (      0 ),
    shape_  (  shape ),
    stride_ ( stride )

{
  index_           = 0_idx;
  index_[LAST_DIM] = shape_[LAST_DIM];
  offset_          = index_[LAST_DIM] * stride_[LAST_DIM];
}


template <class T, int N>

  ArrayIterator<T,N>::ArrayIterator

  ( const Self&  rhs ) noexcept :

    data_   ( rhs.data_   ),
    offset_ ( rhs.offset_ ),
    index_  ( rhs.index_  ),
    shape_  ( rhs.shape_  ),
    stride_ ( rhs.stride_ )

{}


//-----------------------------------------------------------------------
//   increment & decrement operators
//-----------------------------------------------------------------------


template <class T, int N>

  inline ArrayIterator<T,N>&
  ArrayIterator<T,N>::operator ++ ()

{
  offset_ += stride_[0];
  index_[0]++;

  if ( index_[0] >= shape_[0] )
  {
    incrIndex_ ();
  }

  return *this;
}


template <class T, int N>

  inline ArrayIterator<T,N>
  ArrayIterator<T,N>::operator ++ ( int )

{
  Self tmp ( *this );

  operator ++ ();

  return tmp;
}


template <class T, int N>

  inline ArrayIterator<T,N>&
  ArrayIterator<T,N>::operator -- ()

{
  offset_ -= stride_[0];
  index_[0]--;

  if ( index_[0] < 0 )
  {
    decrIndex_ ();
  }

  return *this;
}


template <class T, int N>

  inline ArrayIterator<T,N>
  ArrayIterator<T,N>::operator -- ( int )

{
  Self tmp ( *this );

  operator -- ();

  return tmp;
}


//-----------------------------------------------------------------------
//   access operators
//-----------------------------------------------------------------------


template <class T, int N>

  inline typename ArrayIterator<T,N>::reference
  ArrayIterator<T,N>::operator * () const

{
  return data_ [offset_];
}


template <class T, int N>

  inline typename ArrayIterator<T,N>::pointer
  ArrayIterator<T,N>::operator -> () const

{
  return (data_ + offset_);
}


//-----------------------------------------------------------------------
//   comparison operators
//-----------------------------------------------------------------------


template <class T, int N>

  inline bool
  ArrayIterator<T,N>::operator == ( const Self& rhs ) const noexcept

{
  return (offset_ == rhs.offset_);
}


template <class T, int N>

  inline bool
  ArrayIterator<T,N>::operator != ( const Self& rhs ) const noexcept

{
  return (offset_ != rhs.offset_);
}


//-----------------------------------------------------------------------
//   incrIndex_
//-----------------------------------------------------------------------


template <class T, int N>

  void ArrayIterator<T,N>::incrIndex_ ()

{
  for ( int i = 0; i < LAST_DIM; i++ )
  {
    if ( index_[i] < shape_[i] )
    {
      return;
    }

    index_[i] = 0;
    index_[i + 1]++;
    offset_ -= stride_[i] * shape_[i];
    offset_ += stride_[i + 1];
  }
}


//-----------------------------------------------------------------------
//   decrIndex_
//-----------------------------------------------------------------------


template <class T, int N>

  void ArrayIterator<T,N>::decrIndex_ ()

{
  for ( int i = 0; i < LAST_DIM; i++ )
  {
    if ( index_[i] >= 0 )
    {
      return;
    }

    index_[i] = shape_[i] - 1_idx;
    index_[i + 1]--;
    offset_ += stride_[i] * shape_[i];
    offset_ -= stride_[i + 1];
  }
}


//=======================================================================
//   class ArrayIterator<T,1>
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


template <class T>

  inline ArrayIterator<T,1>::ArrayIterator () noexcept :

    data_   ( nullptr ),
    offset_ ( 0_idx ),
    stride_ ( 1_idx )

{}


template <class T>

  inline ArrayIterator<T,1>::ArrayIterator

  ( T*     data,
    idx_t  offset,
    idx_t  stride ) noexcept :

    data_   ( data ),
    offset_ ( offset ),
    stride_ ( stride )

{}


template <class T>

  inline ArrayIterator<T,1>::ArrayIterator

  ( const Self& rhs ) noexcept :

    data_   ( rhs.data_ ),
    offset_ ( rhs.offset_ ),
    stride_ ( rhs.stride_ )

{}


//-----------------------------------------------------------------------
//   increment & decrement operators
//-----------------------------------------------------------------------


template <class T>

  inline ArrayIterator<T,1>&
  ArrayIterator<T,1>::operator ++ ()

{
  offset_ += stride_;

  return *this;
}


template <class T>

  inline ArrayIterator<T,1>
  ArrayIterator<T,1>::operator ++ ( int )

{
  ArrayIterator tmp ( *this );

  offset_ += stride_;

  return tmp;
}


template <class T>

  inline ArrayIterator<T,1>&
  ArrayIterator<T,1>::operator -- ()

{
  offset_ -= stride_;

  return *this;
}


template <class T>

  inline ArrayIterator<T,1>
  ArrayIterator<T,1>::operator -- ( int )

{
  ArrayIterator tmp ( this );

  offset_ -= stride_;

  return tmp;
}


//-----------------------------------------------------------------------
//   computed assignment operators
//-----------------------------------------------------------------------


template <class T>

  inline ArrayIterator<T,1>&
  ArrayIterator<T,1>::operator += ( idx_t n )

{
  offset_ += n * stride_;

  return *this;
}


template <class T>

  inline ArrayIterator<T,1>&
  ArrayIterator<T,1>::operator -= ( idx_t n )

{
  offset_ -= n * stride_;

  return *this;
}


//-----------------------------------------------------------------------
//   addition & substraction operators
//-----------------------------------------------------------------------


template <class T>

  inline ArrayIterator<T,1>
  ArrayIterator<T,1>::operator + ( idx_t n ) const

{
  return Self ( data_, offset_ + n * stride_, stride_ );
}


template <class T>

  inline ArrayIterator<T,1>
  ArrayIterator<T,1>::operator - ( idx_t n ) const

{
  return Self ( data_, offset_ - n * stride_, stride_ );
}


template <class T>

  inline idx_t
  ArrayIterator<T,1>::operator - ( const Self& rhs ) const

{
  return (offset_ - rhs.offset_) / stride_;
}


//-----------------------------------------------------------------------
//   subscript operator
//-----------------------------------------------------------------------


template <class T>

  inline typename ArrayIterator<T,1>::reference
  ArrayIterator<T,1>::operator [] ( idx_t i ) const

{
  return data_ [offset_ + i * stride_];
}


//-----------------------------------------------------------------------
//   access operators
//-----------------------------------------------------------------------


template <class T>

  inline typename ArrayIterator<T,1>::reference
  ArrayIterator<T,1>::operator * () const

{
  return data_ [offset_];
}


template <class T>

  inline typename ArrayIterator<T,1>::pointer
  ArrayIterator<T,1>::operator -> () const

{
  return data_ + offset_;
}


//-----------------------------------------------------------------------
//   comparison operators
//-----------------------------------------------------------------------


template <class T>

  inline bool
  ArrayIterator<T,1>::operator == ( const Self& rhs ) const noexcept

{
  return (offset_ == rhs.offset_);
}


template <class T>

  inline bool
  ArrayIterator<T,1>::operator != ( const Self& rhs ) const noexcept

{
  return (offset_ != rhs.offset_);
}


template <class T>

  inline bool
  ArrayIterator<T,1>::operator < ( const Self& rhs ) const noexcept

{
  return (offset_ < rhs.offset_);
}


template <class T>

  inline bool
  ArrayIterator<T,1>::operator > ( const Self& rhs ) const noexcept

{
  return (offset_ > rhs.offset_);
}


template <class T>

  inline bool
  ArrayIterator<T,1>::operator <= ( const Self& rhs ) const noexcept

{
  return (offset_ <= rhs.offset_);
}


template <class T>

  inline bool
  ArrayIterator<T,1>::operator >= ( const Self& rhs ) const noexcept

{
  return (offset_ >= rhs.offset_);
}


//=======================================================================
//   related functions
//=======================================================================


template <class T>

  inline idx_t                distance

  ( const ArrayIterator<T,1>&  first,
    const ArrayIterator<T,1>&  last )

{
  return (last - first);
}


JEM_END_PACKAGE_BASE

#endif
