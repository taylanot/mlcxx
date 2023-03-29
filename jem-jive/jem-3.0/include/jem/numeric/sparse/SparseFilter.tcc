
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

#ifndef JEM_NUMERIC_SPARSE_SPARSEFILTER_TCC
#define JEM_NUMERIC_SPARSE_SPARSEFILTER_TCC


JEM_BEGIN_PACKAGE( numeric )


//=======================================================================
//   class SparseFilter<T,PassFilter>
//=======================================================================


template <class T>

  inline bool SparseFilter<T,PassFilter>::accept

  ( const T&, idx_t, idx_t ) const noexcept

{
  return true;
}


//=======================================================================
//   class SparseFilter<T,DiagonalFilter>
//=======================================================================


template <class T>

  inline bool SparseFilter<T,DiagonalFilter>::accept

  ( const T&, idx_t i, idx_t j ) const noexcept

{
  return ( i == j );
}


//=======================================================================
//   class SparseFilter<T,BandFilter>
//=======================================================================


template <class T>

  inline SparseFilter<T,BandFilter>::SparseFilter ( idx_t m ) :

  bandwidth_ ( m )

{}


template <class T>

  inline bool SparseFilter<T,BandFilter>::accept

  ( const T&, idx_t i, idx_t j ) const noexcept

{
  return ( std::abs( i - j ) <= bandwidth_ );
}


//=======================================================================
//   class SparseFilter<T,GreaterFilter>
//=======================================================================


template <class T>

  inline SparseFilter<T,GreaterFilter>::SparseFilter

  ( const T& t ) : threshold_ ( t )

{}


template <class T>

  inline bool SparseFilter<T,GreaterFilter>::accept

  ( const T& value, idx_t, idx_t ) const

{
  return ( value > threshold_ );
}


//=======================================================================
//   class SparseFilter<T,EqualFilter>
//=======================================================================


template <class T>

  inline SparseFilter<T,EqualFilter>::SparseFilter ( const T& t ) :

    value_( t )

{}


template <class T>

  inline bool SparseFilter<T,EqualFilter>::accept

  ( const T& value, idx_t, idx_t ) const

{
  return ( value == value_ );
}


//=======================================================================
//   class SparseFilter<T,UnaryExpr>
//=======================================================================


template <class T, class Op, class F>

  inline SparseFilter<T,UnaryExpr<Op,F> >::SparseFilter

  ( const SparseFilter<T,F>& f ) : filter_ ( f )

{}


template <class T, class Op, class F>

  inline bool SparseFilter<T,UnaryExpr<Op,F> >::accept

  ( const T& value, idx_t i, idx_t j ) const

{
  return op_( filter_.accept( value, i, j ) );
}


//=======================================================================
//   class SparseFilter<BinaryExpr>
//=======================================================================


template <class T, class Op, class F1, class F2>

  inline SparseFilter<T,BinaryExpr<Op,F1,F2> >::SparseFilter

  ( const SparseFilter<T,F1>&  f1,
    const SparseFilter<T,F2>&  f2 ) :

    filter1_ ( f1 ),
    filter2_ ( f2 )

{}


template <class T, class Op, class F1, class F2>

  inline bool SparseFilter<T,BinaryExpr<Op,F1,F2> >::accept

  ( const T& value, idx_t i, idx_t j ) const

{
  return op_( filter1_.accept( value, i, j ),
              filter2_.accept( value, i, j ) );
}


//=======================================================================
//   SparseFilterInterface
//=======================================================================


template <class T>

  inline SparseFilterInterface<T>::~SparseFilterInterface ()

{}


//=======================================================================
//   class SparseFilterWrapper
//=======================================================================


template <class T, class F>

  inline SparseFilterWrapper<T,F>::SparseFilterWrapper ()

{}


template <class T, class F>

  inline SparseFilterWrapper<T,F>::SparseFilterWrapper

  ( const SparseFilter<T,F>& f ) : filter_( f )

{}


template <class T, class F>

  inline SparseFilterWrapper<T,F>::~SparseFilterWrapper ()

{}


template <class T, class F>

  bool SparseFilterWrapper<T,F>::accept

  ( const T& value, idx_t i, idx_t j ) const

{
  return filter_.accept ( value, i, j );
}


//=======================================================================
//   filter operators
//=======================================================================


template <class T, class F>

  inline SparseFilter<T,UnaryExpr<LogicalNot<bool>,F> >

  operator ! ( const SparseFilter<T,F>& f )

{
  return SparseFilter<T,UnaryExpr<LogicalNot<bool>,F> > ( f );
}


template <class T, class F1, class F2>

  inline SparseFilter<T,BinaryExpr<LogicalAnd<bool>,F1,F2> >

  operator && ( const SparseFilter<T,F1>&  lhs,
                const SparseFilter<T,F2>&  rhs )
{
  return SparseFilter<T,BinaryExpr<LogicalAnd<bool>,F1,F2> > (
    lhs,
    rhs
  );
}


template <class T, class F1, class F2>

  inline SparseFilter<T,BinaryExpr<LogicalOr<bool>,F1,F2> >

  operator || ( const SparseFilter<T,F1>&  lhs,
                const SparseFilter<T,F2>&  rhs )

{
  return SparseFilter<T,BinaryExpr<LogicalOr<bool>,F1,F2> > (
    lhs,
    rhs
  );
}


JEM_END_PACKAGE( numeric )

#endif
