
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


#include <jem/base/IllegalArgumentException.h>
#include <jem/hdf5/Library.h>
#include <jem/hdf5/DataSpace.h>
#include <jem/hdf5/H5Exception.h>
#include "Handle.h"


JEM_BEGIN_PACKAGE( hdf5 )


//=======================================================================
//   class DataSpace
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  DataSpace::INVALID_RANK_ERR_   =

  "invalid data space rank";

const char*  DataSpace::INVALID_SHAPE_ERR_  =

  "invalid data space shape";

const char*  DataSpace::INVALID_SELECT_ERR_ =

  "invalid data space selection";


//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


DataSpace::DataSpace ( const Ref<Handle>& sid ) :

  sid_ ( sid )

{
  if ( sid )
  {
    rank_ = H5Sget_simple_extent_ndims ( sid->getID() );

    if ( rank_ < 0 )
    {
      throw H5Exception (
        JEM_FUNC,
        String::format (
          "failed to get the rank of a data space (%s)",
          Library::strerror ()
        )
      );
    }
  }
  else
  {
    rank_ = -1;
  }
}


DataSpace::DataSpace

  ( const idx_t*  shape,
    int           rank )

{
  hid_t  sid;

  if ( rank == 0 )
  {
    sid = H5Screate ( H5S_SCALAR );

    if ( sid < 0 )
    {
      throw H5Exception (
        JEM_FUNC,
        "failed to create a new scalar data space",
        Library::strerror ()
      );
    }
  }
  else
  {
    if ( rank < 6 )
    {
      hsize_t  dims[6];

      for ( int i = 0; i < rank; i++ )
      {
        dims[i] = (hsize_t) shape[i];
      }

      sid = H5Screate_simple ( rank, dims, nullptr );
    }
    else
    {
      Array<hsize_t>  dims ( rank );

      for ( int i = 0; i < rank; i++ )
      {
        dims[i] = (hsize_t) shape[i];
      }

      sid = H5Screate_simple ( rank, dims.addr(), nullptr );
    }

    if ( sid < 0 )
    {
      throw H5Exception (
        JEM_FUNC,
        String::format (
          "failed to create a new data space with rank %d",
          rank
        ),
        Library::strerror ()
      );
    }
  }

  sid_  = newInstance<Handle> ( sid, Handle::ALLOCATED );
  rank_ = rank;
}


//-----------------------------------------------------------------------
//   create
//-----------------------------------------------------------------------


DataSpace DataSpace::create ()
{
  return DataSpace ( nullptr, 0 );
}


DataSpace DataSpace::create ( idx_t n0 )
{
  JEM_PRECHECK2 ( n0 >= 0, INVALID_SHAPE_ERR_);

  return DataSpace ( &n0, 1 );
}


DataSpace DataSpace::create

  ( idx_t  n0,
    idx_t  n1 )

{
  JEM_PRECHECK2 ( n0 >= 0 &&
                  n1 >= 0,
                  INVALID_SHAPE_ERR_ );

  idx_t  shape[2] = { n0, n1 };

  return DataSpace ( shape, 2 );
}


DataSpace DataSpace::create

  ( idx_t  n0,
    idx_t  n1,
    idx_t  n2 )

{
  JEM_PRECHECK2 ( n0 >= 0 &&
                  n1 >= 0 &&
                  n2 >= 0,
                  INVALID_SHAPE_ERR_ );

  idx_t  shape[3] = { n0, n1, n2 };

  return DataSpace ( shape, 3 );
}


DataSpace DataSpace::create

  ( idx_t  n0,
    idx_t  n1,
    idx_t  n2,
    idx_t  n3 )

{
  JEM_PRECHECK2 ( n0 >= 0 &&
                  n1 >= 0 &&
                  n2 >= 0 &&
                  n3 >= 0,
                  INVALID_SHAPE_ERR_ );

  idx_t  shape[4] = { n0, n1, n2, n3 };

  return DataSpace ( shape, 4 );
}


DataSpace DataSpace::create

  ( idx_t  n0,
    idx_t  n1,
    idx_t  n2,
    idx_t  n3,
    idx_t  n4 )

{
  JEM_PRECHECK2 ( n0 >= 0 &&
                  n1 >= 0 &&
                  n2 >= 0 &&
                  n3 >= 0 &&
                  n4 >= 0,
                  INVALID_SHAPE_ERR_ );

  idx_t  shape[5] = { n0, n1, n2, n3, n4 };

  return DataSpace ( shape, 5 );
}


DataSpace DataSpace::create

  ( idx_t  n0,
    idx_t  n1,
    idx_t  n2,
    idx_t  n3,
    idx_t  n4,
    idx_t  n5 )

{
  JEM_PRECHECK2 ( n0 >= 0 &&
                  n1 >= 0 &&
                  n2 >= 0 &&
                  n3 >= 0 &&
                  n4 >= 0 &&
                  n5 >= 0,
                  INVALID_SHAPE_ERR_ );

  idx_t  shape[6] = { n0, n1, n2, n3, n4, n5 };

  return DataSpace ( shape, 6 );
}


DataSpace DataSpace::create

  ( const idx_t*  shape,
    int           rank )

{
  JEM_PRECHECK2 ( rank >= 0, INVALID_RANK_ERR_ );

  for ( int i = 0; i < rank; i++ )
  {
    if ( shape[i] < 0 )
    {
      throw IllegalArgumentException (
        JEM_FUNC,
        "invalid data space shape"
      );
    }
  }

  return DataSpace ( shape, rank );
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


DataSpace DataSpace::clone () const
{
  if ( sid_ )
  {
    hid_t  id = H5Scopy ( sid_->getID() );

    if ( id < 0 )
    {
      throw H5Exception (
        JEM_FUNC,
        String::format (
          "failed to clone a data space (%s)",
          Library::strerror ()
        )
      );
    }

    return DataSpace (
      newInstance<Handle> ( id, Handle::ALLOCATED ),
      rank_
    );
  }
  else
  {
    return *this;
  }
}


//-----------------------------------------------------------------------
//   itemCount
//-----------------------------------------------------------------------


idx_t DataSpace::itemCount () const
{
  JEM_PRECHECK ( isValid() );

  hssize_t  count = H5Sget_select_npoints ( sid_->getID() );

  if ( count < 0 )
  {
    throw H5Exception (
      JEM_FUNC,
      String::format (
        "failed to get the number of selected items "
        "in a data space (%s)",
        Library::strerror ()
      )
    );
  }

  return (idx_t) count;
}


//-----------------------------------------------------------------------
//   getShape
//-----------------------------------------------------------------------


Array<idx_t> DataSpace::getShape () const
{
  JEM_PRECHECK ( isValid() );

  Array<idx_t>  shape ( rank_ );

  getShape ( shape.addr() );

  return shape;
}


void DataSpace::getShape ( idx_t* shape ) const
{
  JEM_PRECHECK ( isValid() );

  const int  rank = rank_;

  if ( rank > 0 )
  {
    int  ndims;

    if ( rank <= 6 )
    {
      hsize_t  dims[6];

      ndims = H5Sget_simple_extent_dims (
        sid_->getID(), dims, nullptr
      );

      if ( ndims == rank )
      {
        for ( int i = 0; i < rank; i++ )
        {
          shape[i] = (idx_t) dims[i];
        }
      }
    }
    else
    {
      Array<hsize_t>  dims ( rank );

      ndims = H5Sget_simple_extent_dims (
        sid_->getID(), dims.addr(), nullptr
      );

      if ( ndims == rank_ )
      {
        for ( int i = 0; i < ndims; i++ )
        {
          shape[i] = (idx_t) dims[i];
        }
      }
    }

    if ( ndims != rank )
    {
      throw H5Exception (
        JEM_FUNC,
        String::format (
          "failed to get the shape of a data space (%s)",
          Library::strerror ()
        )
      );
    }
  }
}


//-----------------------------------------------------------------------
//   select
//-----------------------------------------------------------------------


const DataSpace& DataSpace::select

  ( const Array<idx_t>&  index ) const

{
  JEM_PRECHECK2 ( rank() == 1, INVALID_SELECT_ERR_ );

  if ( index.isContiguous() )
  {
    return selectItems ( index.addr(), index.size() );
  }
  else
  {
    Array<idx_t>  index2 = index.clone ();

    return selectItems ( index2.addr(), index2.size() );
  }
}


const DataSpace& DataSpace::select

  ( const Array<idx_t,2>&  index ) const

{
  JEM_PRECHECK2 ( rank() == index.size(0), INVALID_SELECT_ERR_ );

  if ( index.isContiguous() )
  {
    return selectItems ( index.addr(), index.size() );
  }
  else
  {
    Array<idx_t,2>  index2 = index.clone ();

    return selectItems ( index2.addr(), index2.size(1) );
  }
}


//-----------------------------------------------------------------------
//   selectAll
//-----------------------------------------------------------------------


const DataSpace& DataSpace::selectAll () const
{
  JEM_PRECHECK ( isValid() );

  if ( H5Sselect_all( sid_->getID() ) < 0 )
  {
    throw H5Exception (
      JEM_FUNC,
      "failed to select all array items in a data space",
      Library::strerror ()
    );
  }

  return *this;
}


//-----------------------------------------------------------------------
//   selectSlab
//-----------------------------------------------------------------------


const DataSpace& DataSpace::selectSlab

  ( const idx_t*  offset,
    const idx_t*  stride,
    const idx_t*  count ) const

{
  JEM_PRECHECK ( isValid() );

  const int  rank = rank_;

  if ( rank >= 0 )
  {
    herr_t  status;

    if ( rank <= 6 )
    {
      hsize_t  hoffset[6];
      hsize_t  hstride[6];
      hsize_t  hcount [6];

      for ( int i = 0; i < rank; i++ )
      {
        hoffset[i] = (hsize_t) offset[i];
        hstride[i] = (hsize_t) stride[i];
        hcount [i] = (hsize_t) count [i];
      }

      status = H5Sselect_hyperslab (
        sid_->getID (),
        H5S_SELECT_SET,
        hoffset, hstride, hcount, nullptr
      );
    }
    else
    {
      Array<hsize_t,2>  buffer  ( rank, 3 );

      Array<hsize_t>    hoffset = buffer[0];
      Array<hsize_t>    hstride = buffer[1];
      Array<hsize_t>    hcount  = buffer[2];

      for ( int i = 0; i < rank; i++ )
      {
        hoffset[i] = (hsize_t) offset[i];
        hstride[i] = (hsize_t) stride[i];
        hcount [i] = (hsize_t) count [i];
      }

      status = H5Sselect_hyperslab (
        sid_->getID (),
        H5S_SELECT_SET,
        hoffset.addr(), hstride.addr(), hcount.addr(), nullptr
      );
    }

    if ( status < 0 )
    {
      throw H5Exception (
        JEM_FUNC,
        "failed to select a hyper slab in a data space",
        Library::strerror ()
      );
    }
  }

  return *this;
}


//-----------------------------------------------------------------------
//   selectItems
//-----------------------------------------------------------------------


const DataSpace& DataSpace::selectItems

  ( const idx_t*  index,
    const idx_t   count ) const

{
  JEM_PRECHECK  ( isValid() );
  JEM_PRECHECK2 ( count >= 0, INVALID_SELECT_ERR_ );

  if ( rank_ >= 0 )
  {
    const idx_t     size   = count * rank_;
    Array<hsize_t>  hindex ( size );
    hsize_t*        hptr   = hindex.addr ();

    herr_t          status;

    for ( idx_t i = 0; i < size; i++ )
    {
      hptr[i] = (hsize_t) index[i];
    }

    status = H5Sselect_elements (
      sid_->getID (),
      H5S_SELECT_SET,
      (size_t) count,
      hindex.addr()
    );

    if ( status < 0 )
    {
      throw H5Exception (
        JEM_FUNC,
        "failed to select array items in a data space",
        Library::strerror ()
      );
    }
  }

  return *this;
}


//-----------------------------------------------------------------------
//   select_
//-----------------------------------------------------------------------


void DataSpace::select_

  ( idx_t&          offset,
    idx_t&          stride,
    idx_t&          count,
    idx_t           size,
    const SliceTo&  slice ) const

{
  offset = 0;
  stride = 1;
  count  = slice.last ( size );
}


void DataSpace::select_

  ( idx_t&            offset,
    idx_t&            stride,
    idx_t&            count,
    idx_t             size,
    const SliceFrom&  slice ) const

{
  JEM_PRECHECK2 ( slice.first() <= size, INVALID_SELECT_ERR_ );

  offset = slice.first ();
  stride = 1;
  count  = size - offset;
}


void DataSpace::select_

  ( idx_t&              offset,
    idx_t&              stride,
    idx_t&              count,
    idx_t               size,
    const SliceFromTo&  slice ) const

{
  JEM_PRECHECK2 ( slice.first() <= size,
                  INVALID_SELECT_ERR_ );

  offset = slice.first ();
  stride = 1;
  count  = slice.last( size ) - offset;
}


void DataSpace::select_

  ( idx_t&        offset,
    idx_t&        stride,
    idx_t&        count,
    idx_t         size,
    const Slice&  slice ) const

{
  JEM_PRECHECK2 ( slice.first() <= size, INVALID_SELECT_ERR_ );

  idx_t  last = slice.last ( size );

  offset = slice.first  ();
  stride = slice.stride ();
  count  = last - offset;

  if ( stride > 1_idx )
  {
    count = (count + stride - 1_idx) / stride;
  }
}


JEM_END_PACKAGE( hdf5 )

