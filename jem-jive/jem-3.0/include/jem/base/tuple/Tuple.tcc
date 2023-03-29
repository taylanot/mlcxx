
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

#ifndef JEM_BASE_TUPLE_TUPLE_TCC
#define JEM_BASE_TUPLE_TUPLE_TCC


JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   non-member utility functions
//=======================================================================


JEM_BEGIN_NAMESPACE( tuple )


//-----------------------------------------------------------------------
//   copy
//-----------------------------------------------------------------------


template <class T, class U, class E, int M, int N, int P>

  inline void             copy

  ( Tuple<T,M,N>&           lhs,
    const Tuple<U,M,N,E>&   rhs,
    const IntConstant<P>&   sz )

{
  for ( int i = 0; i < P; i++ )
  {
    lhs[i] = rhs[i];
  }
}


template <class T, class U, class E, int M, int N>

  inline void             copy

  ( Tuple<T,M,N>&           lhs,
    const Tuple<U,M,N,E>&   rhs,
    const IntConstant<0>&   sz )

{}


template <class T, class U, class E, int M, int N>

  inline void             copy

  ( Tuple<T,M,N>&           lhs,
    const Tuple<U,M,N,E>&   rhs,
    const IntConstant<1>&   sz )

{
  lhs[0] = rhs[0];
}


template <class T, class U, class E, int M, int N>

  inline void             copy

  ( Tuple<T,M,N>&           lhs,
    const Tuple<U,M,N,E>&   rhs,
    const IntConstant<2>&   sz )

{
  lhs[0] = rhs[0];
  lhs[1] = rhs[1];
}


template <class T, class U, class E, int M, int N>

  inline void             copy

  ( Tuple<T,M,N>&           lhs,
    const Tuple<U,M,N,E>&   rhs,
    const IntConstant<3>&   sz )

{
  lhs[0] = rhs[0];
  lhs[1] = rhs[1];
  lhs[2] = rhs[2];
}


template <class T, int M, int N, int P>

  inline void             copy

  ( Tuple<T,M,N>&           lhs,
    const Tuple<T,M,N>&     rhs,
    const IntConstant<P>&   sz,
    const True&             pod )

{
  if ( lhs.addr() != rhs.addr() )
  {
    std::memcpy ( lhs.addr(), rhs.addr(), P * sizeof(T) );
  }
}


template <class T, int M, int N>

  inline void             copy

  ( Tuple<T,M,N>&           lhs,
    const Tuple<T,M,N>&     rhs,
    const IntConstant<0>&   sz,
    const True&             pod )

{}


template <class T, int M, int N, int P>

  inline void             copy

  ( Tuple<T,M,N>&           lhs,
    const Tuple<T,M,N>&     rhs,
    const IntConstant<P>&   sz,
    const False&            pod )

{
  for ( int i = 0; i < P; i++ )
  {
    lhs[i] = rhs[i];
  }
}


template <class T, int M, int N>

  inline void             copy

  ( Tuple<T,M,N>&           lhs,
    const Tuple<T,M,N>&     rhs,
    const IntConstant<0>&   sz,
    const False&            pod )

{}


template <class T, int M, int N>

  inline void             copy

  ( Tuple<T,M,N>&           lhs,
    const Tuple<T,M,N>&     rhs,
    const IntConstant<1>&   sz,
    const False&            pod )

{
  lhs[0] = rhs[0];
}


template <class T, int M, int N>

  inline void             copy

  ( Tuple<T,M,N>&           lhs,
    const Tuple<T,M,N>&     rhs,
    const IntConstant<2>&   sz,
    const False&            pod )

{
  lhs[0] = rhs[0];
  lhs[1] = rhs[1];
}


template <class T, int M, int N>

  inline void             copy

  ( Tuple<T,M,N>&           lhs,
    const Tuple<T,M,N>&     rhs,
    const IntConstant<3>&   sz,
    const False&            pod )

{
  lhs[0] = rhs[0];
  lhs[1] = rhs[1];
  lhs[2] = rhs[2];
}


//-----------------------------------------------------------------------
//   fill
//-----------------------------------------------------------------------


template <class T, int M, int N, int P>

  inline void             fill

  ( Tuple<T,M,N>&           lhs,
    const T&                rhs,
    const IntConstant<P>&   sz )

{
  for ( int i = 0; i < P; i++ )
  {
    lhs[i] = rhs;
  }
}


template <class T, int M, int N>

  inline void             fill

  ( Tuple<T,M,N>&           lhs,
    const T&                rhs,
    const IntConstant<0>&   sz )

{}


template <class T, int M, int N>

  inline void             fill

  ( Tuple<T,M,N>&           lhs,
    const T&                rhs,
    const IntConstant<1>&   sz )

{
  lhs[0] = rhs;
}


template <class T, int M, int N>

  inline void             fill

  ( Tuple<T,M,N>&           lhs,
    const T&                rhs,
    const IntConstant<2>&   sz )

{
  lhs[0] = rhs;
  lhs[1] = rhs;
}


template <class T, int M, int N>

  inline void             fill

  ( Tuple<T,M,N>&           lhs,
    const T&                rhs,
    const IntConstant<3>&   sz )

{
  lhs[0] = rhs;
  lhs[1] = rhs;
  lhs[2] = rhs;
}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


using jem::swap;


template <class T, int M, int N, int P>

  inline void             swap

  ( Tuple<T,M,N>&           lhs,
    Tuple<T,M,N>&           rhs,
    const IntConstant<P>&   sz )

{
  for ( int i = 0; i < P; i++ )
  {
    swap ( lhs[i], rhs[i] );
  }
}



template <class T, int M, int N>

  inline void             swap

  ( Tuple<T,M,N>&           lhs,
    Tuple<T,M,N>&           rhs,
    const IntConstant<0>&   sz )

{}


template <class T, int M, int N>

  inline void             swap

  ( Tuple<T,M,N>&           lhs,
    Tuple<T,M,N>&           rhs,
    const IntConstant<1>&   sz )

{
  swap ( lhs[0], rhs[0] );
}


//-----------------------------------------------------------------------
//   equal
//-----------------------------------------------------------------------


template <class T, class U, int M, int N, int P,
          class E1, class E2>

  bool                    equal

  ( const Tuple<T,M,N,E1>&  lhs,
    const Tuple<U,M,N,E2>&  rhs,
    const IntConstant<P>&   sz )

{
  for ( int i = 0; i < P; i++ )
  {
    if ( lhs[i] != rhs[i] )
    {
      return false;
    }
  }

  return true;
}


template <class T, class U, int M, int N, class E1, class E2>

  inline bool             equal

  ( const Tuple<T,M,N,E1>&  lhs,
    const Tuple<U,M,N,E2>&  rhs,
    const IntConstant<0>&   sz )

{
  return true;
}


template <class T, class U, int M, int N, class E1, class E2>

  inline bool             equal

  ( const Tuple<T,M,N,E1>&  lhs,
    const Tuple<U,M,N,E2>&  rhs,
    const IntConstant<1>&   sz )

{
  return (lhs[0] == rhs[0]);
}


template <class T, class U, int M, int N, class E1, class E2>

  inline bool             equal

  ( const Tuple<T,M,N,E1>&  lhs,
    const Tuple<U,M,N,E2>&  rhs,
    const IntConstant<2>&   sz )

{
  return (lhs[0] == rhs[0] &&
          lhs[1] == rhs[1]);
}


template <class T, class U, int M, int N, class E1, class E2>

  inline bool             equal

  ( const Tuple<T,M,N,E1>&  lhs,
    const Tuple<U,M,N,E2>&  rhs,
    const IntConstant<3>&   sz )

{
  return (lhs[0] == rhs[0] &&
          lhs[1] == rhs[1] &&
          lhs[2] == rhs[2]);
}


//-----------------------------------------------------------------------
//   print
//-----------------------------------------------------------------------


template <class Output,
          class T, int M, int N, class E>

  void                    print

  ( Output&                 out,
    const Tuple<T,M,N,E>&   t,
    True                    is1D )

{
  if ( M * N == 0 )
  {
    print ( out, "[]" );
  }
  else
  {
    print ( out, "[ ", t[0] );

    for ( int i = 1; i < M * N; i++ )
    {
      print ( out, ", ", t[i] );
    }

    print ( out, " ]" );
  }
}


template <class Output,
          class T, int M, int N, class E>

  void                    print

  ( Output&                 out,
    const Tuple<T,M,N,E>&   t,
    False                   is1D )

{
  if ( M * N == 0 )
  {
    print ( out, "[[]]" );
  }
  else
  {
    for ( int i = 0; i < M; i++ )
    {
      print ( out, "[ ", t(i,0) );

      for ( int j = 1; j < N; j++ )
      {
        print ( out, ", ", t(i,j) );
      }

      print ( out, " ]" );

      if ( i < M - 1 )
      {
        print ( out, '\n' );
      }
    }
  }
}


JEM_END_NAMESPACE( tuple )


//=======================================================================
//   class Tuple<T,N,Nil>
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


template <class T, int M, int N>

  inline Tuple<T,M,N,Nil>::Tuple ()

{}


template <class T, int M, int N>

  inline Tuple<T,M,N,Nil>::Tuple ( T a )

{
  static_assert ( SIZE == 1, "invalid size" );

  data_[0] = a;
}


template <class T, int M, int N>

  inline Tuple<T,M,N,Nil>::Tuple ( T a, T b )

{
  static_assert ( SIZE == 2, "invalid size" );

  data_[0] = a;
  data_[1] = b;
}


template <class T, int M, int N>

  inline Tuple<T,M,N,Nil>::Tuple ( T a, T b, T c )

{
  static_assert ( SIZE == 3, "invalid size" );

  data_[0] = a;
  data_[1] = b;
  data_[2] = c;
}


template <class T, int M, int N>

  inline Tuple<T,M,N,Nil>::Tuple ( T a, T b, T c, T d )

{
  static_assert ( SIZE == 4, "invalid size" );

  data_[0] = a;
  data_[1] = b;
  data_[2] = c;
  data_[3] = d;
}


template <class T, int M, int N>

  inline Tuple<T,M,N,Nil>::Tuple ( T a, T b, T c, T d, T e )

{
  static_assert ( SIZE == 5, "invalid size" );

  data_[0] = a;
  data_[1] = b;
  data_[2] = c;
  data_[3] = d;
  data_[4] = e;
}


template <class T, int M, int N>

  inline Tuple<T,M,N,Nil>::Tuple ( T a, T b, T c, T d, T e, T f )

{
  static_assert ( SIZE == 6, "invalid size" );

  data_[0] = a;
  data_[1] = b;
  data_[2] = c;
  data_[3] = d;
  data_[4] = e;
  data_[5] = f;
}


template <class T, int M, int N>
  template <class U, class E>

  inline Tuple<T,M,N,Nil>::Tuple ( const Tuple<U,M,N,E>& rhs )

{
  tuple::copy ( *this, rhs, IntConstant<SIZE>() );
}


template <class T, int M, int N>

  inline Tuple<T,M,N,Nil>::Tuple ( const Tuple& rhs )

{
  tuple::copy ( *this, rhs, IntConstant<SIZE>(), IsPOD<T>() );
}


//-----------------------------------------------------------------------
//   initializer list constructor
//-----------------------------------------------------------------------


template <class T, int M, int N>
  template <class U>

  Tuple<T,M,N,Nil>::Tuple ( std::initializer_list<U> list )

{
  JEM_ASSERT2 ( list.size() == SIZE,
                "invalid initializer list size" );

  idx_t  i = 0;

  for ( const U& t : list )
  {
    data_[i++] = t;
  }
}


//-----------------------------------------------------------------------
//   begin & end
//-----------------------------------------------------------------------


template <class T, int M, int N>

  inline T* Tuple<T,M,N,Nil>::begin () noexcept

{
  return data_;
}


template <class T, int M, int N>

  inline const T* Tuple<T,M,N,Nil>::begin () const noexcept

{
  return data_;
}


template <class T, int M, int N>

  inline T* Tuple<T,M,N,Nil>::end () noexcept

{
  return (data_ + SIZE);
}


template <class T, int M, int N>

  inline const T* Tuple<T,M,N,Nil>::end () const noexcept

{
  return (data_ + SIZE);
}


//-----------------------------------------------------------------------
//   assignment operators
//-----------------------------------------------------------------------


template <class T, int M, int N>
  template <class U, class E>

  inline Tuple<T,M,N,Nil>& Tuple<T,M,N,Nil>::operator =

  ( const Tuple<U,M,N,E>& rhs )

{
  tuple::copy ( *this, rhs, IntConstant<SIZE>() );

  return *this;
}


template <class T, int M, int N>

  inline Tuple<T,M,N,Nil>& Tuple<T,M,N,Nil>::operator =

  ( const Tuple& rhs )

{
  tuple::copy ( *this, rhs, IntConstant<SIZE>(), IsPOD<T>() );

  return *this;
}


template <class T, int M, int N>

  inline Tuple<T,M,N,Nil>& Tuple<T,M,N,Nil>::operator =

  ( T rhs )

{
  tuple::fill ( *this, rhs, IntConstant<SIZE>() );

  return *this;
}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


template <class T, int M, int N>

  inline void Tuple<T,M,N,Nil>::swap ( Tuple& rhs )

{
  tuple::swap ( *this, rhs, IntConstant<SIZE>() );
}


//-----------------------------------------------------------------------
//   subscript operators
//-----------------------------------------------------------------------


template <class T, int M, int N>
  template <class I>

  inline T& Tuple<T,M,N,Nil>::operator [] ( I i )

{
  JEM_ASSERT2 ( checkIndex( i, SIZE ), "invalid Tuple index" );

  return data_[i];
}


template <class T, int M, int N>
  template <class I>

  inline const T& Tuple<T,M,N,Nil>::operator [] ( I i ) const

{
  JEM_ASSERT2 ( checkIndex( i, SIZE ), "invalid Tuple index" );

  return data_[i];
}


template <class T, int M, int N>
  template <class I>

  inline T& Tuple<T,M,N,Nil>::operator ()

  ( I i, I j )

{
  JEM_ASSERT2 ( checkIndex( i, M ),
                "invalid Tuple index in dimension 0" );
  JEM_ASSERT2 ( checkIndex( j, N ),
                "invalid Tuple index in dimension 1" );

  return data_[i + j * M];
}


template <class T, int M, int N>
  template <class I>

  inline const T& Tuple<T,M,N,Nil>::operator ()

  ( I i, I j ) const

{
  JEM_ASSERT2 ( checkIndex( i, M ),
                "invalid Tuple index in dimension 0" );
  JEM_ASSERT2 ( checkIndex( j, N ),
               "invalid Tuple index in dimension 1");

  return data_[i + j * M];
}


//-----------------------------------------------------------------------
//   transpose
//-----------------------------------------------------------------------


template <class T, int M, int N>

  Tuple<T,N,M> Tuple<T,M,N,Nil>::transpose () const

{
  Tuple<T,N,M>  t;
  T*            p;

  p = t.addr ();

  for ( int j = 0; j < N; j++ )
  {
    for ( int i = 0; i < M; i++ )
    {
      p[j + i * N] = data_[i + j * M];
    }
  }

  return t;
}


//-----------------------------------------------------------------------
//   addr
//-----------------------------------------------------------------------


template <class T, int M, int N>

  inline T* Tuple<T,M,N,Nil>::addr () noexcept

{
  return data_;
}


template <class T, int M, int N>

  inline const T* Tuple<T,M,N,Nil>::addr () const noexcept

{
  return data_;
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


template <class T, int M, int N>

  inline void swap ( Tuple<T,M,N>& lhs, Tuple<T,M,N>& rhs )

{
  lhs.swap ( rhs );
}


//-----------------------------------------------------------------------
//   equal
//-----------------------------------------------------------------------


template <class T, class U, int M, int N, class E1, class E2>

  inline bool             equal

  ( const Tuple<T,M,N,E1>&  lhs,
    const Tuple<U,M,N,E2>&  rhs )

{
  return tuple::equal ( lhs, rhs, sizeOf(lhs) );
}


//-----------------------------------------------------------------------
//   decode
//-----------------------------------------------------------------------


template <class Input,
          class T, int M, int N>

  inline void           decode

  ( Input&                in,
    Tuple<T,M,N>&         t )

{
  decodeArray ( in, t.addr(), Tuple<T,M,N>::SIZE );
}


//-----------------------------------------------------------------------
//   encode
//-----------------------------------------------------------------------


template <class Output,
          class T, int M, int N>

  inline void           encode

  ( Output&               out,
    const Tuple<T,M,N>&   t )

{
  encodeArray ( out, t.addr(), Tuple<T,M,N>::SIZE );
}


//-----------------------------------------------------------------------
//   print
//-----------------------------------------------------------------------


template <class Output,
          class T, int M, int N, class E>

  inline void             print

  ( Output&                 out,
    const Tuple<T,M,N,E>&   t )

{
  tuple::print ( out, t, BoolConstant<(M == 1 || N == 1)>() );
}


//=======================================================================
//   class TypeTraits< Tuple<T,N> >
//=======================================================================


template <class T, int M, int N>

  class TypeTraits< Tuple<T,M,N> > : public BasicTypeTraits

{
 public:

  static const bool   IS_SERIALIZABLE = IsSerializable<T>::VALUE;


  static String       whatis    ()
  {
    String  kind;

    if ( N == 1 )
    {
      kind = String::format ( " tuple with length %d", M );
    }
    else
    {
      kind = String::format ( " tuple with shape (%d x %d)", M, N );
    }

    return TypeTraits<T>::whatis() + kind;
  }

  static String       typeName  ()
  {
    String  s = TypeTraits<T>::typeName ();

    return String::format ( "jem::Tuple<%s,%d,%d>", s, M, N );
  }

};


JEM_END_PACKAGE_BASE

#endif
