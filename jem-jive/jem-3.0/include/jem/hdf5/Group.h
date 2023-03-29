
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

#ifndef JEM_HDF5_GROUP_H
#define JEM_HDF5_GROUP_H

#include <jem/hdf5/DataSet.h>
#include <jem/hdf5/H5Object.h>


JEM_BEGIN_PACKAGE( hdf5 )


//-----------------------------------------------------------------------
//   class Group
//-----------------------------------------------------------------------


class Group : public H5Object
{
 public:

  JEM_DECLARE_CLASS       ( Group, H5Object );


  explicit                  Group

    ( const Ref<Handle>&      gid );

  void                      addLink

    ( const H5Object&         obj,
      const String&           name )            const;

  bool                      eraseLink

    ( const String&           name )            const;

  bool                      hasGroup

    ( const String&           name )            const;

  Ref<Group>                getGroup

    ( const String&           name )            const;

  Ref<Group>                findGroup

    ( const String&           name )            const;

  Ref<Group>                newGroup

    ( const String&           name )            const;

  bool                      eraseGroup

    ( const String&           name )            const;

  Array<String>             listGroups       () const;
  Array<String>             listAllGroups    () const;

  bool                      hasDataSet

    ( const String&           name )            const;

  Ref<DataSet>              getDataSet

    ( const String&           name )            const;

  template <class T>
    inline void             getDataSet

    ( T&                      data,
      const String&           name )            const;

  template <class T, int N>
    inline void             getDataSet

    ( const Array<T,N>&       data,
      const String&           name )            const;

  Ref<DataSet>              findDataSet

    ( const String&           name )            const;

  template <class T>
    bool                    findDataSet

    ( T&                      data,
      const String&           name )            const;

  template <class T, int N>
    bool                    findDataSet

    ( const Array<T,N>&       data,
      const String&           name )            const;

  Ref<DataSet>              newDataSet

    ( const String&           name,
      const DataType&         type,
      const DataSpace&        space,
      int                     flags = 0 )       const;

  bool                      eraseDataSet

    ( const String&           name )            const;

  template <class T>
    void                    setDataSet

    ( const String&           name,
      const T&                data,
      int                     flags = 0 )       const;

  Array<String>             listDataSets     () const;
  Array<String>             listAllDataSets  () const;


 protected:

  virtual                  ~Group            ();


 private:

  Ref<Handle>               newObject_

    ( const String&           name,
      const DataType*         type  = 0,
      const DataSpace*        space = 0,
      int                     flags = 0 )       const;

  bool                      findObject_

    ( Ref<Handle>*            oid,
      const String&           name )            const;

};




//#######################################################################
//  Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   getDataSet
//-----------------------------------------------------------------------


template <class T>

  inline void Group::getDataSet

  ( T&             data,
    const String&  name ) const

{
  Ref<DataSet>  dset = getDataSet ( name );

  dset->getData ( data );
}


template <class T, int N>

  inline void Group::getDataSet

  ( const Array<T,N>&  data,
    const String&      name ) const

{
  Ref<DataSet>  dset = getDataSet ( name );

  dset->getData ( data );
}


//-----------------------------------------------------------------------
//   findDataSet
//-----------------------------------------------------------------------


template <class T>

  bool Group::findDataSet

  ( T&             data,
    const String&  name ) const

{
  Ref<DataSet>  dset = findDataSet ( name );

  if ( dset )
  {
    dset->getData ( data );

    return true;
  }
  else
  {
    return false;
  }
}


template <class T, int N>

  bool Group::findDataSet

  ( const Array<T,N>&  data,
    const String&      name ) const

{
  Ref<DataSet>  dset = getDataSet ( name );

  if ( dset )
  {
    dset->getData ( data );

    return true;
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------
//   setDataSet
//-----------------------------------------------------------------------


template <class T>

  void Group::setDataSet

  ( const String&  name,
    const T&       data,
    int            flags ) const

{
  Ref<DataSet>  dset = findDataSet ( name );

  if ( ! dset )
  {
    dset = newDataSet ( name,
                        jem::hdf5::typeOf ( data ),
                        jem::hdf5::spaceOf( data ), flags );
  }

  dset->setData ( data );
}


JEM_END_PACKAGE( hdf5 )

#endif
