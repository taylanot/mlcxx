
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

#ifndef JEM_HDF5_CURSOR_H
#define JEM_HDF5_CURSOR_H

#include <jem/base/tuple/Tuple.h>
#include <jem/base/tuple/utilities.h>


JEM_BEGIN_PACKAGE( hdf5 )


//-----------------------------------------------------------------------
//   class Cursor
//-----------------------------------------------------------------------


template <int N>

  class Cursor

{
 public:

  typedef Tuple<idx_t,N>    IdxTuple;


                            Cursor

    ( const IdxTuple&         shape,
      const IdxTuple&         stride );

  idx_t                     next     ();
  inline bool               atEnd    () const noexcept;


 private:

  IdxTuple                  shape_;
  IdxTuple                  stride_;
  IdxTuple                  index_;
  idx_t                     pos_;

};


//-----------------------------------------------------------------------
//   class Cursor<1>
//-----------------------------------------------------------------------


template <>

  class Cursor<1>

{
 public:

  typedef Tuple<idx_t,1>    IdxTuple;


  inline                    Cursor

    ( idx_t                   size,
      idx_t                   stride );

  inline                    Cursor

    ( const IdxTuple&         shape,
      const IdxTuple&         stride );

  inline idx_t              next     ();
  inline bool               atEnd    () const noexcept;


 private:

  idx_t                     end_;
  idx_t                     stride_;
  idx_t                     pos_;

};


//-----------------------------------------------------------------------
//   class Cursor<2>
//-----------------------------------------------------------------------


template <>

  class Cursor<2>

{
 public:

  typedef Tuple<idx_t,2>    IdxTuple;


  inline                    Cursor

    ( const IdxTuple&         shape,
      const IdxTuple&         stride );

  inline idx_t              next     ();
  inline bool               atEnd    () const noexcept;


 private:

  idx_t                     size0_;
  idx_t                     size1_;
  idx_t                     stride0_;
  idx_t                     stride1_;
  idx_t                     index0_;
  idx_t                     index1_;
  idx_t                     pos_;

};



//#######################################################################
//  Implementation
//#######################################################################

//=======================================================================
//   class Cursor
//=======================================================================

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


template <int N>

  Cursor<N>::Cursor

  ( const IdxTuple&  shape,
    const IdxTuple&  stride ) :

    shape_  ( shape ),
    stride_ ( stride )

{
  if ( min( shape ) <= 0_idx )
  {
    shape_ = 0_idx;
  }

  index_ = 0_idx;
  pos_   = 0_idx;
}


//-----------------------------------------------------------------------
//   next
//-----------------------------------------------------------------------


template <int N>

  idx_t Cursor<N>::next ()

{
  idx_t  pos = pos_;

  pos_ += stride_[N - 1];
  index_[N - 1]++;

  if ( index_[N - 1] >= shape_[N - 1] )
  {
    pos_         -= stride_[N - 1] * index_[N - 1];
    index_[N - 1] = 0_idx;
    pos_         += stride_[N - 2];
    index_[N - 2]++;

    for ( int i = N - 2; i > 0; i-- )
    {
      if ( index_[i] < shape_[i] )
      {
        break;
      }
      else
      {
        pos_     -= stride_[i] * index_[i];
        index_[i] = 0_idx;
        pos_     += stride_[i - 1];
        index_[i - 1]++;
      }
    }
  }

  return pos;
}


//-----------------------------------------------------------------------
//   atEnd
//-----------------------------------------------------------------------


template <int N>

  inline bool Cursor<N>::atEnd () const noexcept

{
  return (index_[0] >= shape_[0]);
}


//=======================================================================
//   class Cursor<1>
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline Cursor<1>::Cursor

  ( idx_t  size,
    idx_t  stride )

{
  end_    = size * stride;
  stride_ = stride;
  pos_    = 0_idx;
}


inline Cursor<1>::Cursor

  ( const IdxTuple&  shape,
    const IdxTuple&  stride )

{
  end_    = shape[0] * stride[0];
  stride_ = stride[0];
  pos_    = 0_idx;
}


//-----------------------------------------------------------------------
//   next
//-----------------------------------------------------------------------


inline idx_t Cursor<1>::next ()
{
  idx_t  pos = pos_;

  pos_ += stride_;

  return pos;
}


//-----------------------------------------------------------------------
//   atEnd
//-----------------------------------------------------------------------


inline bool Cursor<1>::atEnd () const noexcept
{
  return (pos_ >= end_);
}


//=======================================================================
//   class Cursor
//=======================================================================

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline Cursor<2>::Cursor

  ( const IdxTuple&  shape,
    const IdxTuple&  stride )

{
  size0_   = shape [0];
  size1_   = shape [1];
  stride0_ = stride[0];
  stride1_ = stride[1];
  index0_  = 0_idx;
  index1_  = 0_idx;
  pos_     = 0_idx;

  if ( (size0_ <= 0_idx) ||
       (size1_ <= 0_idx) )
  {
    size0_ = size1_ = 0_idx;
  }
}


//-----------------------------------------------------------------------
//   next
//-----------------------------------------------------------------------


inline idx_t Cursor<2>::next ()
{
  idx_t  pos = pos_;

  pos_ += stride1_;
  index1_++;

  if ( index1_ >= size1_ )
  {
    pos_   -= stride1_ * index1_;
    index1_ = 0_idx;
    pos_   += stride0_;
    index0_++;
  }

  return pos;
}


//-----------------------------------------------------------------------
//   atEnd
//-----------------------------------------------------------------------


inline bool Cursor<2>::atEnd () const noexcept
{
  return (index0_ >= size0_);
}


JEM_END_PACKAGE( hdf5 )

#endif
