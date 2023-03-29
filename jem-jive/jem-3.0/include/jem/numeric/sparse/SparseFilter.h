
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

#ifndef JEM_NUMERIC_SPARSE_SPARSEFILTER_H
#define JEM_NUMERIC_SPARSE_SPARSEFILTER_H

#include <jem/base/utilities.h>
#include <jem/base/functional.h>
#include <jem/numeric/utilities.h>


JEM_BEGIN_PACKAGE( numeric )


//-----------------------------------------------------------------------
//   class SparseFilter (primary template)
//-----------------------------------------------------------------------


template <class T, class F>  class SparseFilter {};


//-----------------------------------------------------------------------
//   class SparseFilter<T,PassFilter>
//-----------------------------------------------------------------------


template <class T>

  class SparseFilter<T,PassFilter>

{
 public:

  inline bool       accept

    ( const T&        value,
      idx_t           i,
      idx_t           j ) const noexcept;

};


//-----------------------------------------------------------------------
//   class SparseFilter<T,DiagonalFilter>
//-----------------------------------------------------------------------


template <class T>

  class SparseFilter<T,DiagonalFilter>

{
 public:

  inline bool       accept

    ( const T&        value,
      idx_t           i,
      idx_t           j ) const noexcept;

};


//-----------------------------------------------------------------------
//   class SparseFilter<T,BandFilter>
//-----------------------------------------------------------------------


template <class T>

  class SparseFilter<T,BandFilter>

{
 public:

  explicit inline   SparseFilter ( idx_t m );

  inline bool       accept

    ( const T&        value,
      idx_t           i,
      idx_t           j ) const noexcept;


 private:

  idx_t               bandwidth_;

};


//-----------------------------------------------------------------------
//   class SparseFilter<T,GreaterFilter>
//-----------------------------------------------------------------------


template <class T>

  class SparseFilter<T,GreaterFilter>

{
 public:

  explicit inline   SparseFilter ( const T& t );

  inline bool       accept

    ( const T&        value,
      idx_t           i,
      idx_t           j ) const;


 private:

  T                 threshold_;

};


//-----------------------------------------------------------------------
//   class SparseFilter<T,EqualFilter>
//-----------------------------------------------------------------------


template <class T>

  class SparseFilter<T,EqualFilter>

{
 public:

  explicit inline   SparseFilter ( const T& t );

  inline bool       accept

    ( const T&        value,
      idx_t           i,
      idx_t           j ) const;


 private:

  T                 value_;

};


//-----------------------------------------------------------------------
//   class SparseFilter<T,UnaryExpr>
//-----------------------------------------------------------------------


template <class T, class Op, class F>

  class SparseFilter<T,UnaryExpr<Op,F> >

{
 public:

  inline              SparseFilter

    ( const SparseFilter<T,F>&  f );

  inline bool         accept

    ( const T&          value,
      idx_t             i,
      idx_t             j ) const;


 private:

  SparseFilter<T,F>   filter_;
  Op                  op_;

};


//-----------------------------------------------------------------------
//   class SparseFilter<BinaryExpr>
//-----------------------------------------------------------------------


template <class T, class Op, class F1, class F2>

  class SparseFilter<T,BinaryExpr<Op,F1,F2> >

{
 public:

  inline              SparseFilter

    ( const SparseFilter<T,F1>& f1,
      const SparseFilter<T,F2>& f2 );

  inline bool         accept

    ( const T&          value,
      idx_t             i,
      idx_t             j ) const;


 private:

  SparseFilter<T,F1>  filter1_;
  SparseFilter<T,F2>  filter2_;
  Op                  op_;

};


//-----------------------------------------------------------------------
//   class SparseFilterInterface
//-----------------------------------------------------------------------


template <class T>

  class SparseFilterInterface

{
 public:

  virtual inline     ~SparseFilterInterface ();

  virtual bool        accept

    ( const T&          value,
      idx_t             i,
      idx_t             j ) const = 0;

};


//-----------------------------------------------------------------------
//   class SparseFilterWrapper
//-----------------------------------------------------------------------


template <class T, class F>

  class SparseFilterWrapper :
  public SparseFilterInterface<T>

{
 public:

  inline              SparseFilterWrapper ();

  explicit inline     SparseFilterWrapper

    ( const SparseFilter<T,F>&  f );

  virtual inline     ~SparseFilterWrapper ();

  virtual bool        accept

    ( const T&          value,
      idx_t             i,
      idx_t             j ) const override;


 private:

  SparseFilter<T,F>   filter_;

};


//-----------------------------------------------------------------------
//   filter operators
//-----------------------------------------------------------------------


template <class T, class F>

  inline SparseFilter<T,UnaryExpr<LogicalNot<bool>,F> >

  operator !  ( const SparseFilter<T,F>&   f );

template <class T, class F1, class F2>

  inline SparseFilter<T,BinaryExpr<LogicalAnd<bool>,F1,F2> >

  operator && ( const SparseFilter<T,F1>&  lhs,
                const SparseFilter<T,F2>&  rhs );

template <class T, class F1, class F2>

  inline SparseFilter<T,BinaryExpr<LogicalOr<bool>,F1,F2> >

  operator || ( const SparseFilter<T,F1>&  lhs,
                const SparseFilter<T,F2>&  rhs );




//#######################################################################
//   Implementation
//#######################################################################


JEM_END_PACKAGE( numeric )

#include <jem/numeric/sparse/SparseFilter.tcc>

#endif
