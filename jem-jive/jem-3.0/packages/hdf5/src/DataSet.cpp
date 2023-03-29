
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


#include <jem/base/ClassTemplate.h>
#include <jem/hdf5/Library.h>
#include <jem/hdf5/H5Exception.h>
#include <jem/hdf5/DataSet.h>
#include "Handle.h"


JEM_DEFINE_CLASS( jem::hdf5::DataSet );


JEM_BEGIN_PACKAGE( hdf5 )


//=======================================================================
//   class DataSet
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int  DataSet::COMPRESS = 1 << 0;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


DataSet::DataSet ( const Ref<Handle>& oid ) :

  Super ( oid )

{}


DataSet::~DataSet ()
{}


//-----------------------------------------------------------------------
//   itemCount
//-----------------------------------------------------------------------


idx_t DataSet::itemCount () const
{
  return getDataSpace().itemCount ();
}


//-----------------------------------------------------------------------
//   getDataType
//-----------------------------------------------------------------------


DataType DataSet::getDataType () const
{
  hid_t  tid = H5Dget_type ( oid_->getID() );

  if ( tid < 0 )
  {
    throw H5Exception (
      JEM_FUNC,
      String::format (
        "failed to get the type of the %s",
        getContext ()
      ),
      Library::strerror ()
    );
  }

  return DataType (
    newInstance<Handle> ( tid, Handle::ALLOCATED )
  );
}


//-----------------------------------------------------------------------
//   getDataSpace
//-----------------------------------------------------------------------


DataSpace DataSet::getDataSpace () const
{
  hid_t  sid = H5Dget_space ( oid_->getID() );

  if ( sid < 0 )
  {
    throw H5Exception (
      JEM_FUNC,
      String::format (
        "failed to get the data space of the %s",
        getContext ()
      ),
      Library::strerror ()
    );
  }

  return DataSpace (
    newInstance<Handle> ( sid, Handle::ALLOCATED )
  );
}


//-----------------------------------------------------------------------
//   readData
//-----------------------------------------------------------------------


void DataSet::readData

  ( void*            buffer,
    const DataType&  type ) const

{
  JEM_PRECHECK ( type.isValid() );

  herr_t  err = H5Dread ( oid_            ->getID (),
                          type.getHandle()->getID (),
                          H5S_ALL, H5S_ALL,
                          H5P_DEFAULT,
                          buffer );

  if ( err < 0 )
  {
    throw H5Exception (
      JEM_FUNC,
      String::format (
        "error reading data from the %s",
        getContext ()
      ),
      Library::strerror ()
    );
  }
}


void DataSet::readData

  ( void*             buffer,
    const DataType&   type,
    const DataSpace&  mspace,
    const DataSpace&  dspace ) const

{
  JEM_PRECHECK ( type  .isValid() );
  JEM_PRECHECK ( mspace.isValid() );
  JEM_PRECHECK ( dspace.isValid() );

  herr_t  err = H5Dread ( oid_              ->getID (),
                          type  .getHandle()->getID (),
                          mspace.getHandle()->getID (),
                          dspace.getHandle()->getID (),
                          H5P_DEFAULT,
                          buffer );

  if ( err < 0 )
  {
    throw H5Exception (
      JEM_FUNC,
      String::format (
        "error reading data from the %s",
        getContext ()
      ),
      Library::strerror ()
    );
  }
}


//-----------------------------------------------------------------------
//   writeData
//-----------------------------------------------------------------------


void DataSet::writeData

  ( const void*      buffer,
    const DataType&  type ) const

{
  JEM_PRECHECK ( type.isValid() );

  herr_t  err = H5Dwrite ( oid_            ->getID (),
                           type.getHandle()->getID (),
                           H5S_ALL, H5S_ALL,
                           H5P_DEFAULT,
                           buffer );

  if ( err < 0 )
  {
    throw H5Exception (
      JEM_FUNC,
      String::format (
        "error writing data to the %s",
        getContext ()
      ),
      Library::strerror ()
    );
  }
}


void DataSet::writeData

  ( const void*       buffer,
    const DataType&   type,
    const DataSpace&  mspace,
    const DataSpace&  dspace ) const

{
  JEM_PRECHECK ( type  .isValid() );
  JEM_PRECHECK ( mspace.isValid() );
  JEM_PRECHECK ( dspace.isValid() );

  herr_t  err = H5Dwrite ( oid_              ->getID (),
                           type  .getHandle()->getID (),
                           mspace.getHandle()->getID (),
                           dspace.getHandle()->getID (),
                           H5P_DEFAULT,
                           buffer );

  if ( err < 0 )
  {
    throw H5Exception (
      JEM_FUNC,
      String::format (
        "error writing data to the %s",
        getContext ()
      ),
      Library::strerror ()
    );
  }
}


//-----------------------------------------------------------------------
//   countError
//-----------------------------------------------------------------------


void DataSet::countError

  ( const String&  where,
    idx_t          mcount ) const

{
  throw H5Exception (
    where,
    String::format (
      "data count mismatch; expected %d element(s) but "
      "the %s has %d element(s)",
      mcount,
      getContext (),
      itemCount  ()
    )
  );
}


void DataSet::countError

  ( const String&  where,
    idx_t          mcount,
    idx_t          dcount ) const

{
  throw H5Exception (
    where,
    String::format (
      "data count mismatch; the memory buffer has %d element(s) "
      "but %d element(s) have been selected from the %s",
      mcount,
      dcount,
      getContext ()
    )
  );
}


JEM_END_PACKAGE( hdf5 )

