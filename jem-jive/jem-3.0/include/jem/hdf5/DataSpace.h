
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

#ifndef JEM_HDF5_DATASPACE_H
#define JEM_HDF5_DATASPACE_H

#include <jem/base/Ref.h>
#include <jem/base/array/Array.h>


JEM_BEGIN_PACKAGE( hdf5 )


class Handle;


//-----------------------------------------------------------------------
//   class DataSpace
//-----------------------------------------------------------------------


class DataSpace
{
 public:

  inline                    DataSpace   ()       noexcept;

  explicit                  DataSpace

    ( const Ref<Handle>&      sid );

  static DataSpace          create      ();

  static DataSpace          create

    ( idx_t                   n0 );

  static DataSpace          create

    ( idx_t                   n0,
      idx_t                   n1 );

  static DataSpace          create

    ( idx_t                   n0,
      idx_t                   n1,
      idx_t                   n2 );

  static DataSpace          create

    ( idx_t                   n0,
      idx_t                   n1,
      idx_t                   n2,
      idx_t                   n3 );

  static DataSpace          create

    ( idx_t                   n0,
      idx_t                   n1,
      idx_t                   n2,
      idx_t                   n3,
      idx_t                   n4 );

  static DataSpace          create

    ( idx_t                   n0,
      idx_t                   n1,
      idx_t                   n2,
      idx_t                   n3,
      idx_t                   n4,
      idx_t                   n5 );

  static DataSpace          create

    ( const idx_t*            shape,
      int                     rank );

  template <int N>
    static DataSpace        create

    ( const Tuple<idx_t,N>&   shape );

  DataSpace                 clone       () const;
  inline int                rank        () const noexcept;
  inline Handle*            getHandle   () const noexcept;
  inline bool               isValid     () const noexcept;
  idx_t                     itemCount   () const;
  Array<idx_t>              getShape    () const;

  void                      getShape

    ( idx_t*                  shape )      const;

  template <class I>
    const DataSpace&        select

    ( const I&                s0 )         const;

  template <class I,
            class J>
    const DataSpace&        select

    ( const I&               s0,
      const J&               s1 )          const;

  template <class I,
            class J,
            class K>
    const DataSpace&        select

    ( const I&                s0,
      const J&                s1,
      const K&                s2 )         const;

  template <class I,
            class J,
            class K,
            class P>
    const DataSpace&        select

    ( const I&                s0,
      const J&                s1,
      const K&                s2,
      const P&                s3 )         const;

  template <class I,
            class J,
            class K,
            class P,
            class Q>
    const DataSpace&        select

    ( const I&                s0,
      const J&                s1,
      const K&                s2,
      const P&                s3,
      const Q&                s4 )         const;

  template <class I,
            class J,
            class K,
            class P,
            class Q,
            class R>
    const DataSpace&        select

    ( const I&                s0,
      const J&                s1,
      const K&                s2,
      const P&                s3,
      const Q&                s4,
      const R&                s5 )         const;

  template <class I>
    const DataSpace&        select

    ( const Array<I>&         index )      const;

  template <class I>
    const DataSpace&        select

    ( const Array<I,2>&       index )      const;

  const DataSpace&          select

    ( const Array<idx_t>&     index )      const;

  const DataSpace&          select

    ( const Array<idx_t,2>&   index )      const;

  const DataSpace&          selectAll   () const;

  const DataSpace&          selectSlab

    ( const idx_t*            offset,
      const idx_t*            stride,
      const idx_t*            count )      const;

  const DataSpace&          selectItems

    ( const idx_t*            index,
      idx_t                   count )      const;


 private:

  inline                    DataSpace

    ( const Ref<Handle>&      sid,
      int                     rank );

                            DataSpace

    ( const idx_t*            shape,
      int                     rank );

  inline void               select_

    ( idx_t&                  offset,
      idx_t&                  stride,
      idx_t&                  count,
      idx_t                   size,
      idx_t                   index )      const;

  inline void               select_

    ( idx_t&                  offset,
      idx_t&                  stride,
      idx_t&                  count,
      idx_t                   size,
      const SliceAll&         slice )      const;

  void                      select_

    ( idx_t&                  offset,
      idx_t&                  stride,
      idx_t&                  count,
      idx_t                   size,
      const SliceTo&          slice )      const;

  void                      select_

    ( idx_t&                  offset,
      idx_t&                  stride,
      idx_t&                  count,
      idx_t                   size,
      const SliceFrom&        slice )      const;

  void                      select_

    ( idx_t&                  offset,
      idx_t&                  stride,
      idx_t&                  count,
      idx_t                   size,
      const SliceFromTo&      slice )      const;

  void                      select_

    ( idx_t&                  offset,
      idx_t&                  stride,
      idx_t&                  count,
      idx_t                   size,
      const Slice&            slice )      const;


 private:

  static const char*        INVALID_RANK_ERR_;
  static const char*        INVALID_SHAPE_ERR_;
  static const char*        INVALID_SELECT_ERR_;

  Ref<Handle>               sid_;
  int                       rank_;

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


template <class T>

  inline DataSpace          spaceOf

  ( const T&                  obj );

template <class T, int N>

  inline DataSpace          spaceOf

  ( const Array<T,N>&         obj );





//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class DataSpace
//=======================================================================


//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline DataSpace::DataSpace () noexcept
{
  rank_ = -1;
}


inline DataSpace::DataSpace

  ( const Ref<Handle>&  sid,
    int                 rank ) :

    sid_  ( sid ),
    rank_ ( rank )

{}


//-----------------------------------------------------------------------
//   create
//-----------------------------------------------------------------------


template <int N>

  DataSpace DataSpace::create ( const Tuple<idx_t,N>& shape )

{
  JEM_PRECHECK2 ( min( shape ) >= 0, INVALID_SHAPE_ERR_ );

  return DataSpace ( shape.addr(), N );
}


//-----------------------------------------------------------------------
//   rank
//-----------------------------------------------------------------------


inline int DataSpace::rank () const noexcept
{
  return rank_;
}


//-----------------------------------------------------------------------
//   getHandle
//-----------------------------------------------------------------------


inline Handle* DataSpace::getHandle () const noexcept
{
  return sid_.get ();
}


//-----------------------------------------------------------------------
//   isValid
//-----------------------------------------------------------------------


bool DataSpace::isValid () const noexcept
{
  return (sid_ != nullptr);
}


//-----------------------------------------------------------------------
//   select
//-----------------------------------------------------------------------


template <class I>

  const DataSpace& DataSpace::select ( const I& s0 ) const

{
  JEM_PRECHECK2 ( rank() == 1, INVALID_SELECT_ERR_ );

  idx_t  offset;
  idx_t  stride;
  idx_t  count;
  idx_t  shape;

  getShape ( &shape );
  select_  (  offset,  stride,  count, shape, s0 );

  return selectSlab ( &offset, &stride, &count );
}


template <class I,
          class J>

  const DataSpace& DataSpace::select

  ( const I&  s0,
    const J&  s1 ) const

{
  JEM_PRECHECK2 ( rank() == 2, INVALID_SELECT_ERR_ );

  idx_t  offset[2];
  idx_t  stride[2];
  idx_t  count [2];
  idx_t  shape [2];

  getShape ( shape );
  select_  ( offset[0], stride[0], count[0], shape[0], s0 );
  select_  ( offset[1], stride[1], count[1], shape[1], s1 );

  return selectSlab ( offset, stride, count );
}


template <class I,
          class J,
          class K>

  const DataSpace& DataSpace::select

  ( const I&  s0,
    const J&  s1,
    const K&  s2 ) const

{
  JEM_PRECHECK2 ( rank() == 3, INVALID_SELECT_ERR_ );

  idx_t  offset[3];
  idx_t  stride[3];
  idx_t  count [3];
  idx_t  shape [3];

  getShape ( shape );
  select_  ( offset[0], stride[0], count[0], shape[0], s0 );
  select_  ( offset[1], stride[1], count[1], shape[1], s1 );
  select_  ( offset[2], stride[2], count[2], shape[2], s2 );

  return selectSlab ( offset, stride, count );
}


template <class I,
          class J,
          class K,
          class P>

  const DataSpace& DataSpace::select

  ( const I&  s0,
    const J&  s1,
    const K&  s2,
    const P&  s3 ) const

{
  JEM_PRECHECK2 ( rank() == 4, INVALID_SELECT_ERR_ );

  idx_t  offset[4];
  idx_t  stride[4];
  idx_t  count [4];
  idx_t  shape [4];

  getShape ( shape );
  select_  ( offset[0], stride[0], count[0], shape[0], s0 );
  select_  ( offset[1], stride[1], count[1], shape[1], s1 );
  select_  ( offset[2], stride[2], count[2], shape[2], s2 );
  select_  ( offset[3], stride[3], count[3], shape[3], s3 );

  return selectSlab ( offset, stride, count );
}


template <class I,
          class J,
          class K,
          class P,
          class Q>

  const DataSpace& DataSpace::select

  ( const I&  s0,
    const J&  s1,
    const K&  s2,
    const P&  s3,
    const Q&  s4 ) const

{
  JEM_PRECHECK2 ( rank() == 5, INVALID_SELECT_ERR_ );

  idx_t  offset[5];
  idx_t  stride[5];
  idx_t  count [5];
  idx_t  shape [5];

  getShape ( shape );
  select_  ( offset[0], stride[0], count[0], shape[0], s0 );
  select_  ( offset[1], stride[1], count[1], shape[1], s1 );
  select_  ( offset[2], stride[2], count[2], shape[2], s2 );
  select_  ( offset[3], stride[3], count[3], shape[3], s3 );
  select_  ( offset[4], stride[4], count[4], shape[4], s4 );

  return selectSlab ( offset, stride, count );
}


template <class I,
          class J,
          class K,
          class P,
          class Q,
          class R>

  const DataSpace& DataSpace::select

  ( const I&  s0,
    const J&  s1,
    const K&  s2,
    const P&  s3,
    const Q&  s4,
    const R&  s5 ) const

{
  JEM_PRECHECK2 ( rank() == 6, INVALID_SELECT_ERR_ );

  idx_t  offset[6];
  idx_t  stride[6];
  idx_t  count [6];
  idx_t  shape [6];

  getShape ( shape );
  select_  ( offset[0], stride[0], count[0], shape[0], s0 );
  select_  ( offset[1], stride[1], count[1], shape[1], s1 );
  select_  ( offset[2], stride[2], count[2], shape[2], s2 );
  select_  ( offset[3], stride[3], count[3], shape[3], s3 );
  select_  ( offset[4], stride[4], count[4], shape[4], s4 );
  select_  ( offset[5], stride[5], count[5], shape[5], s5 );

  return selectSlab ( offset, stride, count );
}


template <class I>

  const DataSpace& DataSpace::select

  ( const Array<I>&  index ) const

{
  JEM_PRECHECK2 ( rank() == 1, INVALID_SELECT_ERR_ );

  Array<idx_t>  index2 ( index.size() );

  index2 = index;

  return selectItems ( index2.addr(), index2.size() );
}


template <class I>

  const DataSpace& DataSpace::select

  ( const Array<I,2>&  index ) const

{
  JEM_PRECHECK2 ( rank() == index.size(0), INVALID_SELECT_ERR_ );

  Array<idx_t,2>  index2 ( index.shape() );

  index2 = index;

  return selectItems ( index2.addr(), index2.size(1) );
}


//-----------------------------------------------------------------------
//   select_
//-----------------------------------------------------------------------


inline void DataSpace::select_

  ( idx_t&  offset,
    idx_t&  stride,
    idx_t&  count,
    idx_t   size,
    idx_t   index ) const

{
  JEM_PRECHECK2 ( index >= 0 && index < size,
                  INVALID_SELECT_ERR_ );

  offset = index;
  stride = 1;
  count  = 1;
}


inline void DataSpace::select_

  ( idx_t&           offset,
    idx_t&           stride,
    idx_t&           count,
    idx_t            size,
    const SliceAll&  slice ) const

{
  offset = 0;
  stride = 1;
  count  = size;
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   spaceOf
//-----------------------------------------------------------------------


template <class T>

  inline DataSpace          spaceOf

  ( const T&                  obj )

{
  return DataSpace::create ();
}


template <class T, int N>

  inline DataSpace          spaceOf

  ( const Array<T,N>&         obj )

{
  return DataSpace::create ( obj.shape() );
}


JEM_END_PACKAGE( hdf5 )

#endif
