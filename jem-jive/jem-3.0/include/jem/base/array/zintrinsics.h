
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

#ifndef JEM_BASE_ARRAY_ZINTRINSICS_H
#define JEM_BASE_ARRAY_ZINTRINSICS_H

#include <jem/std/zfuncs.h>
#include <jem/base/array/assign.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   class Array<T,N,RealExpr>
//-----------------------------------------------------------------------


template <class T, int N, class E>

  class Array<T,N,RealExpr<E> >

{
 public:

  typedef Tuple<idx_t,N>  Shape;
  typedef T               Reference;
  typedef Array
    <std::complex<T>,N,E> ArgType;


  explicit                Array

    ( const ArgType&        arg ) :

      arg_ ( arg )

  {}

                          Array

    ( const Array&          rhs ) :

      arg_ ( rhs.arg_ )

  {}

  const Array&            operator  =

    ( const Array&          rhs )          const

  {
    array::assign ( arg_, rhs, zfuncs::SetReal<T>() );

    return *this;
  }

  template <class U,
            class E2>
    const Array&          operator =

    ( const Array<U,N,E2>&  rhs )          const

  {
    array::assign ( arg_, rhs, zfuncs::SetReal<T,U>() );

    return *this;
  }

  template <class U>
    const Array&          operator =

    ( const U&              rhs )          const

  {
    array::assign ( arg_, rhs, zfuncs::SetReal<T,U>() );

    return *this;
  }

  template <class U,
            class E2,
            class Op>
    void                  assign

    ( const Array<U,N,E2>&  rhs,
      const Op&             op )           const

  {
    array::assign ( arg_, rhs, zfuncs::SetReal<T,U,Op> ( op ) );
  }

  template <class U,
            class Op>
    void                  assign

    ( const U&              rhs,
      const Op&             op )           const

  {
    array::assign ( arg_, rhs, zfuncs::SetReal<T,U,Op> ( op ) );
  }

  bool                    isContiguous  () const
  {
    return arg_.isContiguous ();
  }

  template <class I>
    Reference             operator []

    ( const I&              i )            const

  {
    return arg_[i].real ();
  }

  Reference               getFast

    ( idx_t                 i )            const

  {
    return arg_.getFast(i).real ();
  }

  idx_t                   size          () const
  {
    return arg_.size ();
  }

  Shape                   shape         () const
  {
    return arg_.shape ();
  }


 private:

  typename ArrayTraits
    <ArgType>::Reference  arg_;

};


//-----------------------------------------------------------------------
//   class Array<T,N,ImagExpr>
//-----------------------------------------------------------------------


template <class T, int N, class E>

  class Array<T,N,ImagExpr<E> >

{
 public:

  typedef Tuple<idx_t,N>  Shape;
  typedef T               Reference;
  typedef Array
    <std::complex<T>,N,E> ArgType;


  explicit                Array

    ( const ArgType&        arg ) :

      arg_ ( arg )

  {}

                          Array

    ( const Array&          rhs ) :

      arg_ ( rhs.arg_ )

  {}

  const Array&            operator  =

    ( const Array&          rhs )          const

  {
    array::assign ( arg_, rhs, zfuncs::SetImag<T>() );

    return *this;
  }

  template <class U,
            class E2>
    const Array&          operator =

    ( const Array<U,N,E2>&  rhs )          const

  {
    array::assign ( arg_, rhs, zfuncs::SetImag<T,U>() );

    return *this;
  }

  template <class U>
    const Array&          operator =

    ( const U&              rhs )          const

  {
    array::assign ( arg_, rhs, zfuncs::SetImag<T,U>() );

    return *this;
  }

  template <class U,
            class E2,
            class Op>
    void                  assign

    ( const Array<U,N,E2>&  rhs,
      const Op&             op )           const

  {
    array::assign ( arg_, rhs, zfuncs::SetImag<T,U,Op> ( op ) );
  }

  template <class U,
            class Op>
    void                  assign

    ( const U&              rhs,
      const Op&             op )           const

  {
    array::assign ( arg_, rhs, zfuncs::SetImag<T,U,Op> ( op ) );
  }

  bool                    isContiguous  () const
  {
    return arg_.isContiguous ();
  }

  template <class I>
    Reference             operator []

    ( const I&              i )            const

  {
    return arg_[i].imag ();
  }

  Reference               getFast

    ( idx_t                 i )            const

  {
    return arg_.getFast(i).imag ();
  }

  idx_t                    size          () const
  {
    return arg_.size ();
  }

  Shape                   shape         () const
  {
    return arg_.shape ();
  }


 private:

  typename ArrayTraits
    <ArgType>::Reference  arg_;

};


//-----------------------------------------------------------------------
//   unary functions
//-----------------------------------------------------------------------

#undef  JEM_DEFINE_UNARY_FUNC
#define JEM_DEFINE_UNARY_FUNC( OP, NAME )           \
                                                    \
template <class T, int N, class E>                  \
  inline Array< typename OP<T>::ResultType, N,      \
    UnaryExpr< OP<T>,                               \
      Array<std::complex<T>,N,E> > >                \
  NAME ( const Array<std::complex<T>,N,E>& e )      \
{                                                   \
  return Array< typename OP<T>::ResultType, N,      \
    UnaryExpr< OP<T>,                               \
      Array<std::complex<T>,N,E> > > ( e );         \
}

JEM_DEFINE_UNARY_FUNC( zfuncs::Abs,  abs  )
JEM_DEFINE_UNARY_FUNC( zfuncs::Arg,  arg  )
JEM_DEFINE_UNARY_FUNC( zfuncs::Conj, conj )
JEM_DEFINE_UNARY_FUNC( zfuncs::Norm, norm )

#undef JEM_DEFINE_UNARY_FUNC


template <class T, int N, class E>

  inline Array<T,N,ImagExpr<E> >

  imag ( const Array<std::complex<T>,N,E>& e )

{
  return Array<T,N,ImagExpr<E> > ( e );
}


template <class T, int N, class E>

  inline Array<T,N,RealExpr<E> >

  real ( const Array<std::complex<T>,N,E>& e )

{
  return Array<T,N,RealExpr<E> > ( e );
}


//-----------------------------------------------------------------------
//   binary functions
//-----------------------------------------------------------------------


#undef  JEM_DEFINE_BINARY_FUNC
#define JEM_DEFINE_BINARY_FUNC( OP, NAME )                \
                                                          \
template <class T, class U, int N, class E>               \
  inline Array< typename OP<T,U>::ResultType, N,          \
    BinaryExpr< OP<T,U>, Array<T,N,E>,                    \
      ScalarExpr<U> > >                                   \
  NAME ( const Array<T,N,E>& lhs, const U& rhs )          \
{                                                         \
  return Array< typename OP<T,U>::ResultType, N,          \
    BinaryExpr< OP<T,U>, Array<T,N,E>,                    \
      ScalarExpr<U> > > ( lhs, rhs );                     \
}                                                         \
                                                          \
template <class T, class U, int N, class E>               \
  inline Array< typename OP<T,U>::ResultType, N,          \
    BinaryExpr< OP<T,U>, ScalarExpr<T>,                   \
      Array<U,N,E> > >                                    \
  NAME ( const T& lhs, const Array<U,N,E>& rhs )          \
{                                                         \
  return Array< typename OP<T,U>::ResultType, N,          \
    BinaryExpr< OP<T,U>, ScalarExpr<T>,                   \
      Array<U,N,E> > > ( lhs, rhs );                      \
}                                                         \
                                                          \
template <class T, class U, int N, class E1, class E2>    \
  inline Array< typename OP<T,U>::ResultType, N,          \
    BinaryExpr< OP<T,U>, Array<T,N,E1>,                   \
      Array<U,N,E2> > >                                   \
  NAME ( const Array<T,N,E1>& lhs,                        \
         const Array<U,N,E2>& rhs )                       \
{                                                         \
  return Array< typename OP<T,U>::ResultType, N,          \
    BinaryExpr< OP<T,U>, Array<T,N,E1>,                   \
      Array<U,N,E2> >  > ( lhs, rhs );                    \
}

JEM_DEFINE_BINARY_FUNC( zfuncs::Zip,   zadd  )
JEM_DEFINE_BINARY_FUNC( zfuncs::Polar, polar )

#undef JEM_DEFINE_BINARY_FUNC

JEM_END_PACKAGE_BASE

#endif
