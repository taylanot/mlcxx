
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


#include <jem/base/StringBuffer.h>
#include <jem/hdf5/Library.h>
#include <jem/hdf5/H5Exception.h>
#include <jem/hdf5/Attribute.h>
#include "Handle.h"


JEM_BEGIN_PACKAGE( hdf5 )


//=======================================================================
//   class Attribute
//=======================================================================

//-----------------------------------------------------------------------
//   itemCount
//-----------------------------------------------------------------------


idx_t Attribute::itemCount () const
{
  return getDataSpace().itemCount ();
}


//-----------------------------------------------------------------------
//   getDataType
//-----------------------------------------------------------------------


DataType Attribute::getDataType () const
{
  JEM_PRECHECK ( isValid() );

  hid_t  tid = H5Aget_type ( aid_->getID() );

  if ( tid < 0 )
  {
    throw H5Exception (
      JEM_FUNC,
      String::format (
        "failed to determine the type of the %s",
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


DataSpace Attribute::getDataSpace () const
{
  JEM_PRECHECK ( isValid() );

  hid_t  sid = H5Aget_space ( aid_->getID() );

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
//   getName
//-----------------------------------------------------------------------


String Attribute::getName () const
{
  JEM_PRECHECK ( isValid() );

  static const char*  UNKNOWN_NAME = "<unknown-name>";

  StringBuffer  buf;
  idx_t         len;


  len = (idx_t) H5Iget_name ( aid_->getID(), nullptr, 0 );

  if ( len < 0 )
  {
    return UNKNOWN_NAME;
  }

  len = (idx_t) H5Iget_name ( aid_->getID (),
                              buf.xalloc  ( len + 1 ),
                              (size_t) len + 1 );

  if ( len < 0 )
  {
    return UNKNOWN_NAME;
  }

  buf.commit ( len );

  buf += '/';
  len  = (idx_t) H5Aget_name ( aid_->getID(), 0, nullptr );

  if ( len < 0 )
  {
    return UNKNOWN_NAME;
  }

  len = (idx_t) H5Aget_name ( aid_->getID (),
                              (size_t) len + 1, buf.xalloc( len + 1 ) );

  if ( len < 0 )
  {
    return UNKNOWN_NAME;
  }

  buf.commit ( len );

  return buf.toString ();
}


//-----------------------------------------------------------------------
//   getContext
//-----------------------------------------------------------------------


String Attribute::getContext  () const
{
  JEM_PRECHECK ( isValid() );

  return String::format ( "HDF5 attribute `%s\'", getName() );
}


//-----------------------------------------------------------------------
//   readData
//-----------------------------------------------------------------------


void Attribute::readData

  ( void*            buffer,
    const DataType&  type ) const

{
  JEM_PRECHECK ( isValid() && type.isValid() );

  herr_t  err = H5Aread ( aid_            ->getID (),
                          type.getHandle()->getID (),
                          buffer );

  if ( err < 0 )
  {
    throw H5Exception (
      getContext (),
      "failed to read attribute data",
      Library::strerror ()
    );
  }
}


//-----------------------------------------------------------------------
//   writeData
//-----------------------------------------------------------------------


void Attribute::writeData

  ( const void*      buffer,
    const DataType&  type ) const

{
  JEM_PRECHECK ( isValid() && type.isValid() );

  herr_t  err = H5Awrite ( aid_            ->getID (),
                           type.getHandle()->getID (),
                           buffer );

  if ( err < 0 )
  {
    throw H5Exception (
      getContext (),
      "failed to write attribute data",
      Library::strerror ()
    );
  }
}


//-----------------------------------------------------------------------
//   countError
//-----------------------------------------------------------------------


void Attribute::countError

  ( const String&  where,
    idx_t          count ) const

{
  throw H5Exception (
    where,
    String::format (
      "data count mismatch; expected %d element(s) but "
      "the %s has %d element(s)",
      count,
      getContext (),
      itemCount  ()
    )
  );
}


JEM_END_PACKAGE( hdf5 )

