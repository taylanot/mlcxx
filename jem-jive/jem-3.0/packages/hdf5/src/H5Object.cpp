
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


#include <jem/base/CString.h>
#include <jem/base/StringBuffer.h>
#include <jem/base/ClassTemplate.h>
#include <jem/hdf5/Library.h>
#include <jem/hdf5/H5Exception.h>
#include <jem/hdf5/DataType.h>
#include <jem/hdf5/DataSpace.h>
#include <jem/hdf5/Attribute.h>
#include <jem/hdf5/H5Object.h>
#include "Handle.h"


JEM_DEFINE_CLASS( jem::hdf5::H5Object );


JEM_BEGIN_PACKAGE( hdf5 )


using util::Flex;


//=======================================================================
//   private functions
//=======================================================================

//-----------------------------------------------------------------------
//   listAttributes_
//-----------------------------------------------------------------------


extern "C"
{
  static herr_t             listAttributes_

    ( hid_t                   oid,
      const char*             name,
      const H5A_info_t*       info,
      void*                   data )

  {
    Flex<String>*  list = (Flex<String>*) data;

    try
    {
      list->pushBack ( clone( name ) );
    }
    catch ( ... )
    {
      return -1;
    }

    return 0;
  }

}


//=======================================================================
//   class H5Object
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


H5Object::H5Object ( const Ref<Handle>& oid ) :

  oid_ ( oid )

{
  JEM_PRECHECK ( oid );
}


H5Object::~H5Object ()
{}


//-----------------------------------------------------------------------
//   getName
//-----------------------------------------------------------------------


String H5Object::getName () const
{
  static const char*  UNKNOWN_NAME;

  StringBuffer  buf;
  idx_t         len;

  len = (idx_t) H5Iget_name ( oid_->getID(), nullptr, 0 );

  if ( len < 0 )
  {
    return UNKNOWN_NAME;
  }

  if ( len == 0 )
  {
    return String ();
  }

  len = (idx_t) H5Iget_name ( oid_->getID (),
                              buf.xalloc  ( len + 1 ),
                              (size_t) len + 1 );

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


String H5Object::getContext () const
{
  H5I_type_t  tid  = H5Iget_type ( oid_->getID() );
  String      name = getName     ();

  String      context;
  String      kind;

  if      ( tid == H5I_FILE )
  {
    kind = "file";
  }
  else if ( tid == H5I_GROUP )
  {
    kind = "group";
  }
  else if ( tid == H5I_DATATYPE )
  {
    kind = "data type";
  }
  else if ( tid == H5I_DATASPACE )
  {
    kind = "data space";
  }
  else if ( tid == H5I_DATASET )
  {
    kind = "data set";
  }
  else if ( tid == H5I_ATTR )
  {
    kind = "attribute";
  }

  if ( name.size() )
  {
    context = String::format (
      "HDF5 %s `%s\'",
      kind,
      name
    );
  }
  else
  {
    context = "anonymous HDF5 " + kind;
  }

  return context;
}


//-----------------------------------------------------------------------
//   hasAttribute
//-----------------------------------------------------------------------


bool H5Object::hasAttribute ( const String& name ) const
{
  htri_t  status = H5Aexists ( oid_->getID (),
                               makeCString ( name ) );

  if ( status < 0 )
  {
    throw H5Exception (
      JEM_FUNC,
      "failed to check for the existence of an HDF5 attribute",
      Library::strerror ()
    );
  }

  return (status > 0);
}


//-----------------------------------------------------------------------
//   getAttribute
//-----------------------------------------------------------------------


Attribute H5Object::getAttribute ( const String& name ) const
{
  hid_t  aid = H5Aopen ( oid_->getID (),
                         makeCString ( name ), H5P_DEFAULT );

  if ( aid < 0 )
  {
    String  details = Library::strerror ();

    if ( ! hasAttribute( name ) )
    {
      throw H5Exception (
        getContext (),
        String::format (
          "attribute `%s\" has not been defined",
          name
        )
      );
    }
    else
    {
      throw H5Exception (
        JEM_FUNC,
        String::format (
          "failed to get the attribute `%s\'",
          name
        ),
        details
      );
    }
  }

  return Attribute (
    newInstance<Handle> ( aid, Handle::ALLOCATED )
  );
}


//-----------------------------------------------------------------------
//   findAttribute
//-----------------------------------------------------------------------


Attribute H5Object::findAttribute ( const String& name ) const
{
  hid_t  aid = H5Aopen ( oid_->getID (),
                         makeCString ( name ), H5P_DEFAULT );

  if ( aid < 0 )
  {
    String  details = Library::strerror ();

    if ( ! hasAttribute( name ) )
    {
      return Attribute ();
    }
    else
    {
      throw H5Exception (
        JEM_FUNC,
        String::format (
          "failed to find the attribute `%s\'",
          name
        ),
        details
      );
    }
  }

  return Attribute (
    newInstance<Handle> ( aid, Handle::ALLOCATED )
  );
}


//-----------------------------------------------------------------------
//   newAttribute
//-----------------------------------------------------------------------


Attribute H5Object::newAttribute

  ( const String&    name,
    const DataType&  type ) const

{
  return newAttribute ( name, type, DataSpace::create() );
}


Attribute H5Object::newAttribute

  ( const String&     name,
    const DataType&   type,
    const DataSpace&  space ) const

{
  JEM_PRECHECK ( type .isValid() );
  JEM_PRECHECK ( space.isValid() );

  hid_t  aid = H5Acreate ( oid_             ->getID (),
                           makeCString        ( name ),
                           type .getHandle()->getID (),
                           space.getHandle()->getID (),
                           H5P_DEFAULT, H5P_DEFAULT );

  if ( aid < 0 )
  {
    throw H5Exception (
      JEM_FUNC,
      String::format (
        "failed to create the attribute `%s\'",
        name
      ),
      Library::strerror ()
    );
  }

  return Attribute (
    newInstance<Handle> ( aid, Handle::ALLOCATED )
  );
}


//-----------------------------------------------------------------------
//   eraseAttribute
//-----------------------------------------------------------------------


void H5Object::eraseAttribute ( const String& name ) const
{
  if ( hasAttribute( name ) )
  {
    herr_t  err = H5Adelete ( oid_->getID(), makeCString( name ) );

    if ( err < 0 )
    {
      throw H5Exception (
        JEM_FUNC,
        String::format (
          "failed to erase the attribute `%s' from the %s",
          name,
          getContext ()
        ),
        Library::strerror ()
      );
    }
  }
}


//-----------------------------------------------------------------------
//   listAttributes
//-----------------------------------------------------------------------


Array<String> H5Object::listAttributes () const
{
  Flex<String>  list;
  hsize_t       pos;
  herr_t        err;


  pos = 0;
  err = H5Aiterate ( oid_->getID (),
                     H5_INDEX_NAME,
                     H5_ITER_INC,
                     &pos,
                     listAttributes_,
                     &list );

  if ( err < 0 )
  {
    throw H5Exception (
      JEM_FUNC,
      String::format (
        "failed to list all attributes of the %s",
        getContext ()
      ),
      Library::strerror ()
    );
  }

  return Array<String> ( list.begin(), list.end() );
}


JEM_END_PACKAGE( hdf5 )

