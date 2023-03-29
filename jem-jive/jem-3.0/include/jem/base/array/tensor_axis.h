
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

#ifndef JEM_BASE_ARRAY_TENSOR_AXIS_H
#define JEM_BASE_ARRAY_TENSOR_AXIS_H

#include <jem/base/functional.h>
#include <jem/base/array/base.h>


JEM_BEGIN_PACKAGE_BASE
JEM_BEGIN_NAMESPACE( tensor )


using jem::max;


//-----------------------------------------------------------------------
//   class Axis
//-----------------------------------------------------------------------


template <class T, class E>

  class Axis {};


//-----------------------------------------------------------------------
//   class Axis<T,UnaryExpr<Op,Axis> >
//-----------------------------------------------------------------------


template <class T, class Op, class U, class E>

  class Axis<T,UnaryExpr<Op,Axis<U,E> > >

{
 public:

  typedef Axis<U,E>       ArgType;


  explicit                Axis

    ( const ArgType&        a ) :

      op  (),
      arg ( a )

  {}

                          Axis

    ( const Axis&           rhs ) :

      op  ( rhs.op  ),
      arg ( rhs.arg )

  {}

  void                    reset         ()
  {
    arg.reset ();
  }

  void                    advance

    ( idx_t*                ix,
      idx_t                 i )

  {
    arg.advance ( ix, i );
  }

  void                    advance

    ( idx_t*                ix,
      idx_t*                jx,
      idx_t                 j )

  {
    arg.advance ( ix, jx, j );
  }

  T                       operator []

    ( idx_t                 i )

  {
    return op ( arg[i] );
  }

  T                       getFast

    ( idx_t                 i )            const

  {
    return op ( arg.getFast(i) );
  }

  idx_t                   getUbound

    ( idx_t*                ix )           const

  {
    return arg.getUbound ( ix );
  }

  void                    checkUbound

    ( idx_t*                ix,
      idx_t                 ub )           const

  {
    arg.checkUbound ( ix, ub );
  }

  bool                    isContiguous  () const
  {
    return arg.isContiguous ();
  }


 private:

  Axis&                   operator =

    ( const Axis&           rhs );


 public:

  const Op                op;
  ArgType                 arg;

};


//-----------------------------------------------------------------------
//   class Axis<T,BinaryExpr<Op,Axis,Axis> >
//-----------------------------------------------------------------------


template <class T, class Op, class U, class V, class E1, class E2>

  class Axis<T,BinaryExpr<Op,Axis<U,E1>,Axis<V,E2> > >

{
 public:

  typedef Axis<U,E1>      Arg1Type;
  typedef Axis<V,E2>      Arg2Type;


                          Axis

    ( const Arg1Type&       a,
      const Arg2Type&       b ) :

      op   (),
      arg1 ( a ),
      arg2 ( b )

  {}

                          Axis

    ( const Axis&           rhs ) :

      op   ( rhs.op   ),
      arg1 ( rhs.arg1 ),
      arg2 ( rhs.arg2 )

  {}

  void                    reset         ()
  {
    arg1.reset ();
    arg2.reset ();
  }

  void                    advance

    ( idx_t*                ix,
      idx_t                 i )

  {
    arg1.advance ( ix, i );
    arg2.advance ( ix, i );
  }

  void                    advance

    ( idx_t*                ix,
      idx_t*                jx,
      idx_t                 j )

  {
    arg1.advance ( ix, jx, j );
    arg2.advance ( ix, jx, j );
  }

  T                       operator []

    ( idx_t                 i )

  {
    return op ( arg1[i], arg2[i] );
  }

  T                       getFast

    ( idx_t                 i )            const

  {
    return op ( arg1.getFast( i ),
                arg2.getFast( i ) );
  }

  idx_t                   getUbound

    ( idx_t*                ix )           const

  {
    return max ( arg1.getUbound( ix ),
                 arg2.getUbound( ix ) );
  }

  void                    checkUbound

    ( idx_t*                ix,
      idx_t                 ub )           const

  {
    arg1.checkUbound ( ix, ub );
    arg2.checkUbound ( ix, ub );
  }

  bool                    isContiguous  () const
  {
    return ( arg1.isContiguous() &&
             arg2.isContiguous() );
  }


 private:

  Axis&                   operator =

    ( const Axis&           rhs );


 public:

  const Op                op;
  Arg1Type                arg1;
  Arg2Type                arg2;

};


//-----------------------------------------------------------------------
//   class Axis<T,BinaryExpr<Op,Axis,Scalar> >
//-----------------------------------------------------------------------


template <class T, class Op, class U, class V, class E>

  class Axis<T,BinaryExpr<Op,Axis<U,E>,ScalarExpr<V> > >

{
 public:

  typedef Axis<U,E>       Arg1Type;
  typedef V               Arg2Type;


                          Axis

    ( const Arg1Type&       a,
      const Arg2Type&       b ) :

      op   (),
      arg1 ( a ),
      arg2 ( b )

  {}

                          Axis

    ( const Axis&           rhs ) :

      op   ( rhs.op   ),
      arg1 ( rhs.arg1 ),
      arg2 ( rhs.arg2 )

  {}

  void                    reset         ()
  {
    arg1.reset ();
  }

  void                    advance

    ( idx_t*                ix,
      idx_t                 i )

  {
    arg1.advance ( ix, i );
   }

  void                    advance

    ( idx_t*                ix,
      idx_t*                jx,
      idx_t                 j )

  {
    arg1.advance ( ix, jx, j );
  }

  T                       operator []

    ( idx_t                 i )

  {
    return op ( arg1[i], arg2 );
  }

  T                       getFast

    ( idx_t                 i )            const

  {
    return op ( arg1.getFast( i ), arg2 );
  }

  idx_t                   getUbound

    ( idx_t*                ix )           const

  {
    return arg1.getUbound ( ix );
  }

  void                    checkUbound

    ( idx_t*                ix,
      idx_t                 ub )           const

  {
    arg1.checkUbound ( ix, ub );
  }

  bool                    isContiguous  () const
  {
    return arg1.isContiguous ();
  }


 private:

  Axis&                   operator =

    ( const Axis&           rhs );


 public:

  const Op                op;
  Arg1Type                arg1;
  const Arg2Type          arg2;

};


//-----------------------------------------------------------------------
//   class Axis<T,BinaryExpr<Op,Scalar,Axis> >
//-----------------------------------------------------------------------


template <class T, class Op, class U, class V, class E>

  class Axis<T,BinaryExpr<Op,ScalarExpr<U>,Axis<V,E> > >

{
 public:

  typedef U               Arg1Type;
  typedef Axis<V,E>       Arg2Type;


                          Axis

    ( const Arg1Type&       a,
      const Arg2Type&       b ) :

      op   (),
      arg1 ( a ),
      arg2 ( b )

  {}

                          Axis

    ( const Axis&           rhs ) :

      op   ( rhs.op   ),
      arg1 ( rhs.arg1 ),
      arg2 ( rhs.arg2 )

  {}

  void                    reset         ()
  {
    arg2.reset ();
  }

  void                    advance

    ( idx_t*                ix,
      idx_t                 i )

  {
    arg2.advance ( ix, i );
  }

  void                    advance

    ( idx_t*                ix,
      idx_t*                jx,
      idx_t                 j )

  {
    arg2.advance ( ix, jx, j );
  }

  T                       operator []

    ( idx_t                 i )

  {
    return op ( arg1, arg2[i] );
  }

  T                       getFast

    ( idx_t                 i )            const

  {
    return op ( arg1, arg2.getFast( i ) );
  }

  idx_t                   getUbound

    ( idx_t*                ix )           const

  {
    return arg2.getUbound ( ix );
  }

  void                    checkUbound

    ( idx_t*                ix,
      idx_t                 ub )           const

  {
    arg2.checkUbound ( ix, ub );
  }

  bool                    isContiguous  () const
  {
    return arg2.isContiguous ();
  }


 private:

  Axis&                   operator =

    ( const Axis&           rhs );


 public:

  const Op                op;
  const Arg1Type          arg1;
  Arg2Type                arg2;

};


//-----------------------------------------------------------------------
//   class Axis<T,WhereExpr<Axis,Axis,Axis> >
//-----------------------------------------------------------------------


template <class T, class U, class V, class E1, class E2, class E3>

  class Axis<T,WhereExpr<Axis<bool,E1>,Axis<U,E2>,Axis<V,E3> > >

{
 public:

  typedef Axis<bool,E1>   MaskType;
  typedef Axis<U,E2>      Arg1Type;
  typedef Axis<V,E3>      Arg2Type;


                          Axis

    ( const MaskType&       a,
      const Arg1Type&       b,
      const Arg2Type&       c ) :

      mask ( a ),
      arg1 ( b ),
      arg2 ( c )

  {}

                          Axis

    ( const Axis&           rhs ) :

      mask ( rhs.mask ),
      arg1 ( rhs.arg1 ),
      arg2 ( rhs.arg2 )

  {}

  void                    reset         ()
  {
    mask.reset ();
    arg1.reset ();
    arg2.reset ();
  }

  void                    advance

    ( idx_t*                ix,
      idx_t                 i )

  {
    mask.advance ( ix, i );
    arg1.advance ( ix, i );
    arg2.advance ( ix, i );
  }

  void                    advance

    ( idx_t*                ix,
      idx_t*                jx,
      idx_t                 j )

  {
    mask.advance ( ix, jx, j );
    arg1.advance ( ix, jx, j );
    arg2.advance ( ix, jx, j );
  }

  T                       operator []

    ( idx_t                 i )

  {
    return ( mask[i] ? arg1[i] : arg2[i] );
  }

  T                       getFast

    ( idx_t                 i )            const

  {
    return ( mask.getFast( i ) ?
             arg1.getFast( i ) :
             arg2.getFast( i ) );
  }

  idx_t                   getUbound

    ( idx_t*                ix )           const

  {
    return max ( mask.getUbound( ix ),
                 arg1.getUbound( ix ),
                 arg2.getUbound( ix ) );
  }

  void                    checkUbound

    ( idx_t*                ix,
      idx_t                 ub )           const

  {
    mask.checkUbound ( ix, ub );
    arg1.checkUbound ( ix, ub );
    arg2.checkUbound ( ix, ub );
  }

  bool                    isContiguous  () const
  {
    return ( mask.isContiguous() &&
             arg1.isContiguous() &&
             arg2.isContiguous() );
  }


 private:

  Axis&                   operator =

    ( const Axis&           rhs );


 public:

  MaskType                mask;
  Arg1Type                arg1;
  Arg2Type                arg2;

};


//-----------------------------------------------------------------------
//   class Axis<T,WhereExpr<Axis,Axis,Scalar> >
//-----------------------------------------------------------------------


template <class T, class U, class V, class E1, class E2>

  class Axis<T,WhereExpr<Axis<bool,E1>,
    Axis<U,E2>, ScalarExpr<V> > >

{
 public:

  typedef Axis<bool,E1>   MaskType;
  typedef Axis<U,E2>      Arg1Type;
  typedef V               Arg2Type;


                          Axis

    ( const MaskType&       a,
      const Arg1Type&       b,
      const Arg2Type&       c ) :

      mask ( a ),
      arg1 ( b ),
      arg2 ( c )

  {}

                          Axis

    ( const Axis&           rhs ) :

      mask ( rhs.mask ),
      arg1 ( rhs.arg1 ),
      arg2 ( rhs.arg2 )

  {}

  void                    reset         ()
  {
    mask.reset ();
    arg1.reset ();
  }

  void                    advance

    ( idx_t*                ix,
      idx_t                 i )

  {
    mask.advance ( ix, i );
    arg1.advance ( ix, i );
  }

  void                    advance

    ( idx_t*                ix,
      idx_t*                jx,
      idx_t                 j )

  {
    mask.advance ( ix, jx, j );
    arg1.advance ( ix, jx, j );
  }

  T                       operator []

    ( idx_t                 i )

  {
    return ( mask[i] ? arg1[i] : arg2 );
  }

  T                       getFast

    ( idx_t                 i )            const

  {
    return ( mask.getFast( i ) ?
             arg1.getFast( i ) : arg2 );
  }

  idx_t                   getUbound

    ( idx_t*                ix )           const

  {
    return max ( mask.getUbound( ix ),
                 arg1.getUbound( ix ) );
  }

  void                    checkUbound

    ( idx_t*                ix,
      idx_t                 ub )           const

  {
    mask.checkUbound ( ix, ub );
    arg1.checkUbound ( ix, ub );
  }

  bool                    isContiguous  () const
  {
    return ( mask.isContiguous() &&
             arg1.isContiguous() );
  }


 private:

  Axis&                   operator =

    ( const Axis&           rhs );


 public:

  MaskType                mask;
  Arg1Type                arg1;
  const Arg2Type          arg2;

};


//-----------------------------------------------------------------------
//   class Axis<T,WhereExpr<Axis,Scalar,Axis> >
//-----------------------------------------------------------------------


template <class T, class U, class V, class E1, class E2>

  class Axis<T,WhereExpr<Axis<bool,E1>,
    ScalarExpr<U>, Axis<V,E2> > >

{
 public:

  typedef Axis<bool,E1>   MaskType;
  typedef U               Arg1Type;
  typedef Axis<V,E2>      Arg2Type;


                          Axis

    ( const MaskType&       a,
      const Arg1Type&       b,
      const Arg2Type&       c ) :

      mask ( a ),
      arg1 ( b ),
      arg2 ( c )

  {}

                          Axis

    ( const Axis&           rhs ) :

      mask ( rhs.mask ),
      arg1 ( rhs.arg1 ),
      arg2 ( rhs.arg2 )

  {}

  void                    reset         ()
  {
    mask.reset ();
    arg2.reset ();
  }

  void                    advance

    ( idx_t*                ix,
      idx_t                 i )

  {
    mask.advance ( ix, i );
    arg2.advance ( ix, i );
  }

  void                    advance

    ( idx_t*                ix,
      idx_t*                jx,
      idx_t                 j )

  {
    mask.advance ( ix, jx, j );
    arg2.advance ( ix, jx, j );
  }

  T                       operator []

    ( idx_t                 i )

  {
    return ( mask[i] ? arg1 : arg2[i] );
  }

  T                       getFast

    ( idx_t                 i )            const

  {
    return ( mask.getFast( i ) ? arg1 :
             arg2.getFast( i ) );
  }

  idx_t                   getUbound

    ( idx_t*                ix )           const

  {
    return max ( mask.getUbound( ix ),
                 arg2.getUbound( ix ) );
  }

  void                    checkUbound

    ( idx_t*                ix,
      idx_t                 ub )           const

  {
    mask.checkUbound ( ix, ub );
    arg2.checkUbound ( ix, ub );
  }

  bool                    isContiguous  () const
  {
    return ( mask.isContiguous() &&
             arg2.isContiguous() );
  }


 private:

  Axis&                   operator =

    ( const Axis&           rhs );


 public:

  MaskType                mask;
  const Arg1Type          arg1;
  Arg2Type                arg2;

};


//-----------------------------------------------------------------------
//   class Axis<T,WhereExpr<Axis,Scalar,Scalar> >
//-----------------------------------------------------------------------


template <class T, class U, class V, class E>

  class Axis<T,WhereExpr<Axis<bool,E>,
    ScalarExpr<U>, ScalarExpr<V> > >

{
 public:

  typedef Axis<bool,E>    MaskType;
  typedef U               Arg1Type;
  typedef V               Arg2Type;


                          Axis

    ( const MaskType&       a,
      const Arg1Type&       b,
      const Arg2Type&       c ) :

      mask ( a ),
      arg1 ( b ),
      arg2 ( c )

  {}

                          Axis

    ( const Axis&           rhs ) :

      mask ( rhs.mask ),
      arg1 ( rhs.arg1 ),
      arg2 ( rhs.arg2 )

  {}

  void                    reset         ()
  {
    mask.reset ();
  }

  void                    advance

    ( idx_t*                ix,
      idx_t                 i )

  {
    mask.advance ( ix, i );
  }

  void                    advance

    ( idx_t*                ix,
      idx_t*                jx,
      idx_t                 j )

  {
    mask.advance ( ix, jx, j );
  }

  T                       operator []

    ( idx_t                 i )

  {
    return ( mask[i] ? arg1 : arg2 );
  }

  T                       getFast

    ( idx_t                 i )            const

  {
    return ( mask.getFast( i ) ? arg1 : arg2 );
  }

  idx_t                   getUbound

    ( idx_t*                ix )           const

  {
    return mask.getUbound ( ix );
  }

  void                    checkUbound

    ( idx_t*                ix,
      idx_t                 ub )           const

  {
    mask.checkUbound ( ix, ub );
  }

  bool                    isContiguous  () const
  {
    return mask.isContiguous ();
  }


 private:

  Axis&                   operator =

    ( const Axis&           rhs );


 public:

  MaskType                mask;
  const Arg1Type          arg1;
  const Arg2Type          arg2;

};


//-----------------------------------------------------------------------
//   class AxisTraits
//-----------------------------------------------------------------------


template <class T>

  class AxisTraits {};


//-----------------------------------------------------------------------
//   class AxisTraits<Scalar>
//-----------------------------------------------------------------------


template <class T>

  class AxisTraits<ScalarExpr<T> >

{
 public:

  typedef ScalarExpr<T>   AxisType;

  static const T&         getAxis

    ( const T&              a,
      idx_t*                ix,
      idx_t*                jx )

  {
    return a;
  }

};


//-----------------------------------------------------------------------
//   class AxisTraits<Array>
//-----------------------------------------------------------------------


template <class T, class E>

  class AxisTraits< Array<T,0,E> >

{
 public:

  typedef Axis<T,E>       AxisType;

  static AxisType         getAxis

    ( const Array<T,0,E>&   a,
      idx_t*                ix,
      idx_t*                jx )

  {
    return AxisType ( a, ix, jx );
  }

};


//-----------------------------------------------------------------------
//   class AxisTraits<Array<T,0,UnaryExpr> >
//-----------------------------------------------------------------------


template <class T, class Op, class A>

  class AxisTraits< Array<T,0,UnaryExpr<Op,A> > >

{
 public:

  typedef typename AxisTraits<A>::AxisType  Arg1Type;
  typedef Axis<T,UnaryExpr<Op,Arg1Type> >   AxisType;

  static AxisType         getAxis

    ( const Array<T,0,
      UnaryExpr<Op,A> >&    a,
      idx_t*                ix,
      idx_t*                jx )

  {
    return AxisType ( AxisTraits<A>::getAxis( a.arg, ix, jx ) );
  }

};


//-----------------------------------------------------------------------
//   class AxisTraits<Array<T,0,BinaryExpr> >
//-----------------------------------------------------------------------


template <class T, class Op, class A, class B>

  class AxisTraits< Array<T,0,BinaryExpr<Op,A,B> > >

{
 public:

  typedef typename AxisTraits<A>::AxisType  Arg1Type;
  typedef typename AxisTraits<B>::AxisType  Arg2Type;
  typedef Axis
    <T,BinaryExpr<Op,Arg1Type,Arg2Type> >   AxisType;

  static AxisType         getAxis

    ( const Array<T,0,
      BinaryExpr<Op,A,B> >& a,
      idx_t*                ix,
      idx_t*                jx )

  {
    return AxisType ( AxisTraits<A>::getAxis( a.arg1, ix, jx ),
                      AxisTraits<B>::getAxis( a.arg2, ix, jx ) );
  }

};


//-----------------------------------------------------------------------
//   class AxisTraits<Array<T,0,WhereExpr> >
//-----------------------------------------------------------------------


template <class T, class A, class B, class C>

  class AxisTraits< Array<T,0,WhereExpr<A,B,C> > >

{
 public:

  typedef typename AxisTraits<A>::AxisType  MaskType;
  typedef typename AxisTraits<B>::AxisType  Arg1Type;
  typedef typename AxisTraits<C>::AxisType  Arg2Type;
  typedef Axis<T,WhereExpr
    <MaskType,Arg1Type,Arg2Type> >          AxisType;

  static AxisType         getAxis

    ( const Array<T,0,
      WhereExpr<A,B,C> >&   a,
      idx_t*                ix,
      idx_t*                jx )

  {
    return AxisType ( AxisTraits<A>::getAxis( a.mask, ix, jx ),
                      AxisTraits<B>::getAxis( a.arg1, ix, jx ),
                      AxisTraits<C>::getAxis( a.arg2, ix, jx ) );
  }

};


JEM_END_NAMESPACE( tensor )


//-----------------------------------------------------------------------
//   getAxis
//-----------------------------------------------------------------------


template <class T, class E>

  inline typename tensor::AxisTraits< Array<T,0,E> >::AxisType

                        getAxis

  ( const Array<T,0,E>&   a,
    idx_t*                ix,
    idx_t*                jx )

{
  return tensor::AxisTraits< Array<T,0,E> >::getAxis ( a, ix, jx );
}


JEM_END_PACKAGE_BASE

#endif
