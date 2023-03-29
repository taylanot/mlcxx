
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
#include <jem/base/ClassTemplate.h>
#include <jem/hdf5/Library.h>
#include <jem/hdf5/H5Exception.h>
#include <jem/hdf5/DataSet.h>
#include <jem/hdf5/DataSpace.h>
#include <jem/hdf5/Group.h>
#include "Handle.h"


JEM_DEFINE_CLASS( jem::hdf5::Group );


JEM_BEGIN_PACKAGE( hdf5 )


using util::Flex;


//=======================================================================
//   private functions
//=======================================================================

//-----------------------------------------------------------------------
//   listGroups_
//-----------------------------------------------------------------------


extern "C"
{
  static herr_t             listGroups_

    ( hid_t                   gid,
      const char*             name,
      const H5L_info_t*       info,
      void*                   data )

  {
    Flex<String>*  list = (Flex<String>*) data;

    H5O_info_t     info2;
    herr_t         err;


    try
    {
      err = H5Oget_info_by_name ( gid, name, &info2, H5P_DEFAULT );

      if ( err < 0 )
      {
        return err;
      }

      if ( info2.type == H5O_TYPE_GROUP )
      {
        list->pushBack ( clone( name ) );
      }
    }
    catch ( ... )
    {
      return -1;
    }

    return 0;
  }

}


//-----------------------------------------------------------------------
//   listAllGroups_
//-----------------------------------------------------------------------


extern "C"
{
  static herr_t             listAllGroups_

    ( hid_t                   gid,
      const char*             name,
      const H5O_info_t*       info,
      void*                   data )

  {
    Flex<String>*  list = (Flex<String>*) data;

    try
    {
      if ( info->type == H5O_TYPE_GROUP )
      {
        String  str = clone ( name );

        if ( str != "." )
        {
          list->pushBack ( std::move( str ) );
        }
      }
    }
    catch ( ... )
    {
      return -1;
    }

    return 0;
  }

}


//-----------------------------------------------------------------------
//   listDataSets_
//-----------------------------------------------------------------------


extern "C"
{
  static herr_t             listDataSets_

    ( hid_t                   gid,
      const char*             name,
      const H5L_info_t*       info,
      void*                   data )

  {
    Flex<String>*  list = (Flex<String>*) data;

    H5O_info_t     info2;
    herr_t         err;


    try
    {
      err = H5Oget_info_by_name ( gid, name, &info2, H5P_DEFAULT );

      if ( err < 0 )
      {
        return err;
      }

      if ( info2.type == H5O_TYPE_DATASET )
      {
        list->pushBack ( clone( name ) );
      }
    }
    catch ( ... )
    {
      return -1;
    }

    return 0;
  }

}


//-----------------------------------------------------------------------
//   listAllDataSets_
//-----------------------------------------------------------------------


extern "C"
{
  static herr_t             listAllDataSets_

    ( hid_t                   gid,
      const char*             name,
      const H5O_info_t*       info,
      void*                   data )

  {
    Flex<String>*  list = (Flex<String>*) data;

    try
    {
      if ( info->type == H5O_TYPE_DATASET )
      {
        String  str = clone ( name );

        if ( str != "." )
        {
          list->pushBack ( std::move( str ) );
        }
      }
    }
    catch ( ... )
    {
      return -1;
    }

    return 0;
  }

}


//=======================================================================
//   class Group
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


Group::Group ( const Ref<Handle>& oid ) :

  Super ( oid )

{}


Group::~Group ()
{}


//-----------------------------------------------------------------------
//   addLink
//-----------------------------------------------------------------------


void Group::addLink

  ( const H5Object&  obj,
    const String&    name ) const

{
  hid_t   gid = this->getHandle()->getID ();
  hid_t   oid = obj  .getHandle()->getID ();
  hid_t   pid = -1;
  herr_t  err;


  pid = H5Pcreate ( H5P_LINK_CREATE );

  if ( pid >= 0 )
  {
    H5Pset_create_intermediate_group ( pid, 1 );

    err = H5Olink ( oid, gid, makeCString( name ),
                    pid, H5P_DEFAULT );

    H5Pclose ( pid );
  }
  else
  {
    err = H5Olink ( oid, gid, makeCString( name ),
                    H5P_DEFAULT, H5P_DEFAULT );
  }

  if ( err < 0 )
  {
    throw H5Exception (
      getContext (),
      String::format (
        "failed to create the HDF5 link `%s\'",
        name
      ),
      Library::strerror ()
    );
  }
}


//-----------------------------------------------------------------------
//   eraseLink
//-----------------------------------------------------------------------


bool Group::eraseLink ( const String& name ) const
{
  if ( findObject_( nullptr, name ) )
  {
    herr_t  err = H5Ldelete ( oid_->getID (),
                              makeCString ( name ),
                              H5P_DEFAULT );

    if ( err < 0 )
    {
      throw H5Exception (
        getContext (),
        String::format (
          "failed to erase the HDF5 link `%s\'",
          name
        ),
        Library::strerror ()
      );
    }

    return true;
  }

  return false;
}


//-----------------------------------------------------------------------
//   hasGroup
//-----------------------------------------------------------------------


bool Group::hasGroup ( const String& name ) const
{
  Ref<Handle>  gid;

  if ( findObject_( &gid, name ) )
  {
    H5I_type_t  type = H5Iget_type ( gid->getID() );

    if ( type == H5I_GROUP )
    {
      return true;
    }
  }

  return false;
}


//-----------------------------------------------------------------------
//   getGroup
//-----------------------------------------------------------------------


Ref<Group> Group::getGroup ( const String& name ) const
{
  hid_t  gid = H5Gopen ( oid_->getID (),
                         makeCString ( name ),
                         H5P_DEFAULT );

  if ( gid < 0 )
  {
    String  details = Library::strerror ();

    if ( hasGroup( name ) )
    {
      throw H5Exception (
        getContext (),
        String::format (
          "failed to open the HDF5 group `%s\'",
          name
        ),
        details
      );throw H5Exception (
        getContext (),
        String::format (
          "failed to open the HDF5 group `%s\'",
          name
        ),
        details
      );
    }
    else
    {
      throw H5Exception (
        getContext (),
        String::format (
          "the HDF5 group `%s\' does not exist",
          name
        )
      );
    }
  }

  return newInstance<Group> (
    newInstance<Handle> ( gid, Handle::ALLOCATED )
  );
}


//-----------------------------------------------------------------------
//   findGroup
//-----------------------------------------------------------------------


Ref<Group> Group::findGroup ( const String& name ) const
{
  hid_t  gid = H5Gopen ( oid_->getID (),
                         makeCString ( name ),
                         H5P_DEFAULT );

  if ( gid < 0 )
  {
    String  details = Library::strerror ();

    if ( hasGroup( name ) )
    {
      throw H5Exception (
        getContext (),
        String::format (
          "failed to open the HDF5 group `%s\'",
          name
        ),
        details
      );
    }
    else
    {
      return nullptr;
    }
  }

  return newInstance<Group> (
    newInstance<Handle> ( gid, Handle::ALLOCATED )
  );
}


//-----------------------------------------------------------------------
//   newGroup
//-----------------------------------------------------------------------


Ref<Group> Group::newGroup ( const String& name ) const
{
  eraseGroup ( name );

  return newInstance<Group> ( newObject_( name ) );
}


//-----------------------------------------------------------------------
//   eraseGroup
//-----------------------------------------------------------------------


bool Group::eraseGroup ( const String& name ) const
{
  if ( hasGroup( name ) )
  {
    herr_t  err = H5Ldelete ( oid_->getID (),
                              makeCString ( name ),
                              H5P_DEFAULT );

    if ( err < 0 )
    {
      throw H5Exception (
        getContext (),
        String::format (
          "failed to erase the HDF5 group `%s\'",
          name
        ),
        Library::strerror ()
      );
    }

    return true;
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------
//   listGroups
//-----------------------------------------------------------------------


Array<String> Group::listGroups () const
{
  Flex<String>  list;
  herr_t        err;
  hsize_t       pos;

  pos = 0;
  err = H5Literate ( oid_->getID (),
                     H5_INDEX_NAME,
                     H5_ITER_INC,
                     &pos,
                     listGroups_,
                     &list );

  if ( err < 0 )
  {
    throw H5Exception (
      getContext (),
      String::format (
        "failed to list the sub groups of the %s",
        getContext ()
      ),
      Library::strerror ()
    );
  }

  return Array<String> ( list.begin(), list.end() );
}


//-----------------------------------------------------------------------
//   listAllGroups
//-----------------------------------------------------------------------


Array<String> Group::listAllGroups () const
{
  Flex<String>  list;
  herr_t        err;

  err = H5Ovisit ( oid_->getID (),
                   H5_INDEX_CRT_ORDER,
                   H5_ITER_INC,
                   listAllGroups_,
                   &list );

  if ( err < 0 )
  {
    throw H5Exception (
      getContext (),
      String::format (
        "failed to recursively list all sub groups of the %s",
        getContext ()
      ),
      Library::strerror ()
    );
  }

  return Array<String> ( list.begin(), list.end() );
}


//-----------------------------------------------------------------------
//   hasDataSet
//-----------------------------------------------------------------------


bool Group::hasDataSet ( const String& name ) const
{
  Ref<Handle>  sid;

  if ( findObject_( &sid, name ) )
  {
    H5I_type_t  type = H5Iget_type ( sid->getID() );

    if ( type == H5I_DATASET )
    {
      return true;
    }
  }

  return false;
}


//-----------------------------------------------------------------------
//   getDataSet
//-----------------------------------------------------------------------


Ref<DataSet> Group::getDataSet ( const String& name ) const
{
  hid_t  sid = H5Dopen ( oid_->getID (),
                         makeCString ( name ),
                         H5P_DEFAULT );

  if ( sid < 0 )
  {
    String  details = Library::strerror ();

    if ( hasDataSet( name ) )
    {
      throw H5Exception (
        getContext (),
        String::format (
          "failed to open the HDF5 data set `%s\'",
          name
        ),
        details
      );
    }
    else
    {
      throw H5Exception (
        getContext (),
        String::format (
          "the HDF5 data set `%s\' does not exist",
          name
        )
      );
    }
  }

  return newInstance<DataSet> (
    newInstance<Handle> ( sid, Handle::ALLOCATED )
  );
}


//-----------------------------------------------------------------------
//   findDataSet
//-----------------------------------------------------------------------


Ref<DataSet> Group::findDataSet ( const String& name ) const
{
  hid_t  sid = H5Dopen ( oid_->getID (),
                         makeCString ( name ),
                         H5P_DEFAULT );

  if ( sid < 0 )
  {
    String  details = Library::strerror ();

    if ( hasDataSet( name ) )
    {
      throw H5Exception (
        getContext (),
        String::format (
          "failed to open the HDF5 data set `%s\'",
          name
        ),
        details
      );
    }
    else
    {
      return nullptr;
    }
  }

  return newInstance<DataSet> (
    newInstance<Handle> ( sid, Handle::ALLOCATED )
  );
}


//-----------------------------------------------------------------------
//   newDataSet
//-----------------------------------------------------------------------


Ref<DataSet> Group::newDataSet

  ( const String&     name,
    const DataType&   type,
    const DataSpace&  space,
    int               flags ) const

{
  JEM_PRECHECK ( type .isValid() );
  JEM_PRECHECK ( space.isValid() );

  eraseDataSet ( name );

  return newInstance<DataSet> ( newObject_( name,  &type,
                                            &space, flags ) );
}


//-----------------------------------------------------------------------
//   eraseDataSet
//-----------------------------------------------------------------------


bool Group::eraseDataSet ( const String& name ) const
{
  if ( hasDataSet( name ) )
  {
    herr_t  err = H5Ldelete ( oid_->getID (),
                              makeCString ( name ),
                              H5P_DEFAULT );

    if ( err < 0 )
    {
      throw H5Exception (
        getContext (),
        String::format (
          "failed to erase the HDF5 data set `%s\'",
          name
        ),
        Library::strerror ()
      );
    }

    return true;
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------
//   listDataSets
//-----------------------------------------------------------------------


Array<String> Group::listDataSets () const
{
  Flex<String>  list;
  herr_t        err;
  hsize_t       pos;

  pos = 0;
  err = H5Literate ( oid_->getID (),
                     H5_INDEX_NAME,
                     H5_ITER_INC,
                     &pos,
                     listDataSets_,
                     &list );

  if ( err < 0 )
  {
    throw H5Exception (
      getContext (),
      String::format (
        "failed to list the data sets in the %s",
        getContext ()
      ),
      Library::strerror ()
    );
  }

  return Array<String> ( list.begin(), list.end() );
}


//-----------------------------------------------------------------------
//   listAllDataSets
//-----------------------------------------------------------------------


Array<String> Group::listAllDataSets () const
{
  Flex<String>  list;
  herr_t        err;

  err = H5Ovisit ( oid_->getID (),
                   H5_INDEX_CRT_ORDER,
                   H5_ITER_INC,
                   listAllDataSets_,
                   &list );

  if ( err < 0 )
  {
    throw H5Exception (
      getContext (),
      String::format (
        "failed to recursively list all data sets in the %s",
        getContext ()
      ),
      Library::strerror ()
    );
  }

  return Array<String> ( list.begin(), list.end() );
}


//-----------------------------------------------------------------------
//   newObject_
//-----------------------------------------------------------------------


Ref<Handle> Group::newObject_

  ( const String&     name,
    const DataType*   type,
    const DataSpace*  space,
    int               flags ) const

{
  hid_t   pid = H5P_DEFAULT;
  htri_t  stat;
  hid_t   oid;
  idx_t   i, j;


  i = 0;

  if ( name.startsWith( '/' ) )
  {
    j = name.find ( '/', 1 );
  }
  else
  {
    j = name.find ( '/' );
  }

  while ( j >= 0 )
  {
    if ( (j - i) > 1 || name[i] != '/' )
    {
      stat = H5Lexists ( oid_->getID (),
                         makeCString ( name[slice(BEGIN,j)] ),
                         H5P_DEFAULT );

      if ( stat < 0 )
      {
        goto error;
      }

      if ( ! stat )
      {
        hid_t  gid =

          H5Gcreate ( oid_->getID (),
                      makeCString ( name[slice(BEGIN,j)] ),
                      H5P_DEFAULT,
                      H5P_DEFAULT,
                      H5P_DEFAULT );

        if ( gid < 0 )
        {
          throw H5Exception (
            getContext (),
            String::format (
              "failed to create the HDF5 group `%s\'",
              name
            ),
            Library::strerror ()
          );
        }
      }
    }

    i = j;
    j = name.find ( '/', j + 1 );
  }

  if ( (name.size() - i) == 0 || name.endsWith( '/' ) )
  {
    goto error;
  }

  if ( space )
  {
    JEM_PRECHECK ( type );

    if ( flags )
    {
      pid = H5Pcreate ( H5P_DATASET_CREATE );

      if ( pid < 0 )
      {
        goto error;
      }

      if ( (flags & DataSet::COMPRESS) &&
           (space->rank() > 0) && ! type->hasVarSize() )
      {
        const int       rank  = space->rank     ();
        Array<idx_t>    shape = space->getShape ();
        Array<hsize_t>  dims  ( rank );

        for ( int i = 0; i < rank; i++ )
        {
          dims[i] = (hsize_t) shape[i];
        }

        // Switch to CHUNKED layout; this is required when using
        // a compression filter.

        if ( H5Pset_layout( pid, H5D_CHUNKED ) < 0 )
        {
          goto error;
        }

        if ( H5Pset_chunk( pid, rank, dims.addr() ) < 0 )
        {
          goto error;
        }

        if ( H5Pset_filter( pid, H5Z_FILTER_DEFLATE,
                                 H5Z_FLAG_OPTIONAL, 0, nullptr ) < 0 )
        {
          goto error;
        }

        if ( H5Pset_deflate( pid, 6 ) < 0 )
        {
          goto error;
        }
      }
    }

    if ( name.size() )
    {
      oid = H5Dcreate ( oid_->getID (),
                        makeCString ( name ),
                        type ->getHandle()->getID (),
                        space->getHandle()->getID (),
                        H5P_DEFAULT,
                        pid,
                        H5P_DEFAULT );
    }
    else
    {
      oid = H5Dcreate_anon ( oid_              ->getID (),
                             type ->getHandle()->getID (),
                             space->getHandle()->getID (),
                             pid,
                             H5P_DEFAULT );

    }
  }
  else
  {
    if ( name.size() )
    {
      oid = H5Gcreate ( oid_->getID (),
                        makeCString ( name ),
                        H5P_DEFAULT,
                        H5P_DEFAULT,
                        H5P_DEFAULT );
    }
    else
    {
      oid = H5Gcreate_anon ( oid_->getID (),
                             H5P_DEFAULT,
                             H5P_DEFAULT );
    }
  }

  if ( pid >= 0 && pid != H5P_DEFAULT )
  {
    H5Pclose ( pid );
  }

  if ( oid >= 0 )
  {
    return newInstance<Handle> ( oid, Handle::ALLOCATED );
  }

error:

  if ( pid >= 0 && pid != H5P_DEFAULT )
  {
    H5Pclose ( pid );
  }

  if ( space )
  {
    throw H5Exception (
      getContext (),
      String::format (
        "failed to create the HDF5 data set `%s\'",
        name
      ),
      Library::strerror ()
    );
  }
  else
  {
    throw H5Exception (
      getContext (),
      String::format (
        "failed to create the HDF5 group `%s\'",
        name
      ),
      Library::strerror ()
    );
  }

  return nullptr;
}


//-----------------------------------------------------------------------
//   findObject_
//-----------------------------------------------------------------------


bool Group::findObject_

  ( Ref<Handle>*   oid,
    const String&  name ) const

{
  const idx_t  len = name.size ();

  htri_t       stat;
  idx_t        i, j;

  if ( len == 0 )
  {
    return false;
  }

  i = 0;

  if ( name.startsWith( '/' ) )
  {
    j = name.find ( '/', 1 );
  }
  else
  {
    j = name.find ( '/' );
  }

  while ( i < len )
  {
    if ( j < 0 )
    {
      j = len;
    }

    if ( (j - i) > 1 || name[i] != '/' )
    {
      stat = H5Lexists ( oid_->getID (),
                         makeCString ( name[slice(BEGIN,j)] ),
                         H5P_DEFAULT );

      if ( stat < 0 )
      {
        goto error;
      }

      if ( ! stat )
      {
        return false;
      }

      stat = H5Oexists_by_name ( oid_->getID (),
                                 makeCString ( name[slice(BEGIN,j)] ),
                                 H5P_DEFAULT );

      if ( stat < 0 )
      {
        goto error;
      }

      if ( ! stat )
      {
        return false;
      }
    }

    i = j;

    if ( i < len )
    {
      j = name.find ( '/', i + 1 );
    }
  }

  if ( oid )
  {
    hid_t  id = H5Oopen ( oid_->getID (),
                          makeCString ( name ),
                          H5P_DEFAULT );

    if ( id < 0 )
    {
      goto error;
    }

    *oid = newInstance<Handle> ( id, Handle::ALLOCATED );
  }

  return true;

error:

  throw H5Exception (
    getContext (),
    String::format (
      "failed to determine the existence of the "
      "HDF5 object `%s\'",
      name
    ),
    Library::strerror ()
  );

  return false;
}


JEM_END_PACKAGE( hdf5 )

