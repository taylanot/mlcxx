
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

#ifndef JEM_BASE_ARRAY_TENSOR_INDEX_H
#define JEM_BASE_ARRAY_TENSOR_INDEX_H

#include <jem/base/array/base.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   class Array<idx_t,0,IndexExpr>
//-----------------------------------------------------------------------


template <>

  class Array<idx_t,0,IndexExpr>

{
 public:

  inline                  Array         ();

  explicit inline         Array

    ( const Slice&          rhs );

  inline                  Array

    ( idx_t                 first,
      idx_t                 last,
      idx_t                 stride = 1 );

  inline                  Array

    ( idx_t                 first,
      const End&            last,
      idx_t                 stride = 1 );

  inline                  Array

    ( const SliceAll&       all,
      idx_t                 stride = 1 );

  inline                  Array

    ( const Array&          rhs );

  inline                  Array

    ( const Array&          rhs,
      idx_t                 first,
      idx_t                 last,
      idx_t                 stride );

  inline idx_t*           addr          () const;
  inline idx_t            first         () const;
  inline idx_t            last          () const;
  inline idx_t            last

    ( idx_t                 ubound )       const;

  inline idx_t            stride        () const;
  inline bool             isSimple      () const;


 private:

  Array&                  operator      =

    ( const Array&          rhs );

  inline void             init_

    ( idx_t*                addr,
      idx_t                 first,
      idx_t                 last,
      idx_t                 stride );


 private:

  idx_t*                  addr_;
  idx_t                   value_;
  idx_t                   first_;
  idx_t                   last_;
  idx_t                   stride_;
  bool                    simple_;

};


//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline Array<idx_t,0,IndexExpr>::Array ()
{
  addr_   = &value_;
  value_  = -1_idx;
  first_  =  0_idx;
  last_   =  Slice::MAX_INDEX;
  stride_ =  1_idx;
  simple_ =  true;
}


inline Array<idx_t,0,IndexExpr>::Array ( const Slice& rhs )
{
  init_ ( &value_, rhs.first(), rhs.last(), rhs.stride() );
}


inline Array<idx_t,0,IndexExpr>::Array

  ( idx_t  first,
    idx_t  last,
    idx_t  stride )

{
  init_ ( &value_, first, last, stride );
}


inline Array<idx_t,0,IndexExpr>::Array

  ( idx_t       first,
    const End&  last,
    idx_t       stride )

{
  init_ ( &value_, first, Slice::MAX_INDEX, stride );
}


inline Array<idx_t,0,IndexExpr>::Array

  ( const SliceAll&  all,
    idx_t            stride )

{
  init_ ( &value_, 0, Slice::MAX_INDEX, stride );
}


inline Array<idx_t,0,IndexExpr>::Array ( const Array& rhs )
{
  addr_   = rhs.addr_;
  value_  = -1_idx;
  first_  = rhs.first_;
  last_   = rhs.last_;
  stride_ = rhs.stride_;
  simple_ = rhs.simple_;
}


inline Array<idx_t,0,IndexExpr>::Array

  ( const Array&  rhs,
    idx_t         first,
    idx_t         last,
    idx_t         stride )

{
  init_ ( rhs.addr_, first, last, stride );
}


//-----------------------------------------------------------------------
//   addr
//-----------------------------------------------------------------------


inline idx_t* Array<idx_t,0,IndexExpr>::addr () const
{
  return addr_;
}


//-----------------------------------------------------------------------
//   first
//-----------------------------------------------------------------------


inline idx_t Array<idx_t,0,IndexExpr>::first () const
{
  return first_;
}


//-----------------------------------------------------------------------
//   last
//-----------------------------------------------------------------------


inline idx_t Array<idx_t,0,IndexExpr>::last () const
{
  return last_;
}


inline idx_t Array<idx_t,0,IndexExpr>::last ( idx_t ubound ) const
{
  return ( (ubound < last_) ? ubound : last_ );
}


//-----------------------------------------------------------------------
//   stride
//-----------------------------------------------------------------------


inline idx_t Array<idx_t,0,IndexExpr>::stride () const
{
  return stride_;
}


//-----------------------------------------------------------------------
//   isSimple
//-----------------------------------------------------------------------


inline bool Array<idx_t,0,IndexExpr>::isSimple () const
{
  return simple_;
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


inline void Array<idx_t,0,IndexExpr>::init_

  ( idx_t*  addr,
    idx_t   first,
    idx_t   last,
    idx_t   stride )

{
  addr_   = addr;
  value_  = -1_idx;
  first_  = first;
  last_   = last;
  stride_ = stride;
  simple_ = false;
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   operator +
//-----------------------------------------------------------------------


template <class I>

  inline TensorIndex    operator +

  ( const TensorIndex&    lhs,
    I                     rhs )

{
  idx_t  irhs = (idx_t) rhs;
  idx_t  itmp = lhs.last ();
  idx_t  last = (itmp == Slice::MAX_INDEX) ? itmp : itmp + irhs;

  return TensorIndex ( lhs, lhs.first() + irhs, last, lhs.stride() );
}


template <class I>

  inline TensorIndex    operator +

  ( I                     lhs,
    const TensorIndex&    rhs )

{
  return operator + ( rhs, lhs );
}


//-----------------------------------------------------------------------
//   operator -
//-----------------------------------------------------------------------


template <class I>

  inline TensorIndex    operator -

  ( const TensorIndex&    lhs,
    I                     rhs )

{
  idx_t  irhs = (idx_t) rhs;
  idx_t  itmp = lhs.last ();
  idx_t  last = (itmp == Slice::MAX_INDEX) ? itmp : itmp - irhs;

  return TensorIndex ( lhs, lhs.first() - irhs, last, lhs.stride() );
}


//-----------------------------------------------------------------------
//   operator *
//-----------------------------------------------------------------------


template <class I>

  inline TensorIndex    operator *

  ( const TensorIndex&    lhs,
    I                     rhs )

{
  idx_t  irhs = (idx_t) rhs;
  idx_t  itmp = lhs.last ();
  idx_t  last = (itmp == Slice::MAX_INDEX) ? itmp : itmp * irhs;

  return TensorIndex ( lhs,  lhs.first()  * irhs,
                       last, lhs.stride() * irhs );
}


template <class I>

  inline TensorIndex    operator *

  ( I                     lhs,
    const TensorIndex&    rhs )

{
  return operator * ( rhs, lhs );
}


//-----------------------------------------------------------------------
//   operator /
//-----------------------------------------------------------------------


template <class I>

  inline TensorIndex    operator /

  ( const TensorIndex&    lhs,
    I                     rhs )

{
  JEM_ASSERT2 ( rhs != 0, "tensor index divided by zero" );

  idx_t  irhs = (idx_t) rhs;
  idx_t  itmp = lhs.last ();
  idx_t  last = (itmp == Slice::MAX_INDEX) ? itmp : itmp / irhs;

  return TensorIndex ( lhs,  lhs.first()  / irhs,
                       last, lhs.stride() / irhs );
}


JEM_END_PACKAGE_BASE

#endif
