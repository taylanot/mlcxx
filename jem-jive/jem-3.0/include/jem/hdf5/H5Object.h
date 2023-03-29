
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

#ifndef JEM_HDF5_H5OBJECT_H
#define JEM_HDF5_H5OBJECT_H

#include <jem/base/Object.h>
#include <jem/hdf5/DataType.h>
#include <jem/hdf5/Attribute.h>


JEM_BEGIN_PACKAGE( hdf5 )


class Handle;


//-----------------------------------------------------------------------
//   class H5Object
//-----------------------------------------------------------------------


class H5Object : public Object
{
 public:

  JEM_DECLARE_CLASS       ( H5Object, Object );


  inline Handle*            getHandle       () const noexcept;
  String                    getName         () const;
  String                    getContext      () const;

  bool                      hasAttribute

    ( const String&           name )           const;

  Attribute                 getAttribute

    ( const String&           name )           const;

  template <class T>
    inline void             getAttribute

    ( T&                      value,
      const String&           name )           const;

  template <class T, int N>
    inline void             getAttribute

    ( const Array<T,N>&       value,
      const String&           name )           const;

  Attribute                 findAttribute

    ( const String&           name )           const;

  template <class T>
    bool                    findAttribute

    ( T&                      value,
      const String&           name )           const;

  template <class T, int N>
    bool                    findAttribute

    ( const Array<T,N>&       value,
      const String&           name )           const;

  Attribute                 newAttribute

    ( const String&           name,
      const DataType&         type )           const;

  Attribute                 newAttribute

    ( const String&           name,
      const DataType&         type,
      const DataSpace&        space )          const;

  template <class T>
    void                    setAttribute

    ( const String&           name,
      const T&                value )          const;

  void                      eraseAttribute

    ( const String&           name )           const;

  Array<String>             listAttributes  () const;


 protected:

  explicit                  H5Object

    ( const Ref<Handle>&      oid );

  virtual                  ~H5Object        ();


 protected:

  Ref<Handle>               oid_;

};



//#######################################################################
//  implementation
//#######################################################################

//-----------------------------------------------------------------------
//   getHandle
//-----------------------------------------------------------------------


inline Handle* H5Object::getHandle () const noexcept
{
  return oid_.get ();
}


//-----------------------------------------------------------------------
//   getAttribute
//-----------------------------------------------------------------------


template <class T>

  inline void H5Object::getAttribute

  ( T&             value,
    const String&  name ) const

{
  Attribute  attr = getAttribute ( name );

  attr.getValue ( value );
}


template <class T, int N>

  inline void H5Object::getAttribute

  ( const Array<T,N>&  value,
    const String&      name ) const

{
  Attribute  attr = getAttribute ( name );

  attr.getValue ( value );
}


//-----------------------------------------------------------------------
//   findAttribute
//-----------------------------------------------------------------------


template <class T>

  bool H5Object::findAttribute

  ( T&             value,
    const String&  name ) const

{
  Attribute  attr = findAttribute ( name );

  if ( attr.isValid() )
  {
    attr.getValue ( value );

    return true;
  }
  else
  {
    return false;
  }
}


template <class T, int N>

  bool H5Object::findAttribute

  ( const Array<T,N>&  value,
    const String&      name ) const

{
  Attribute  attr = getAttribute ( name );

  if ( attr.isValid() )
  {
    attr.getValue ( value );

    return true;
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------
//   setAttribute
//-----------------------------------------------------------------------


template <class T>

  void H5Object::setAttribute

  ( const String&  name,
    const T&       value ) const

{
  Attribute  attr = findAttribute ( name );

  if ( ! attr.isValid() )
  {
    attr = newAttribute ( name,
                          jem::hdf5::typeOf( value ),
                          jem::hdf5::spaceOf( value ) );
  }

  attr.setValue ( value );
}


JEM_END_PACKAGE( hdf5 )

#endif
