
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

#ifndef JEM_BASE_STRINGOBJECT_H
#define JEM_BASE_STRINGOBJECT_H

#include <jem/base/Object.h>
#include <jem/base/Clonable.h>
#include <jem/io/Serializable.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   class StringObject
//-----------------------------------------------------------------------


class StringObject : public Object,
                     public Clonable,
                     public io::Serializable
{
 public:

  JEM_DECLARE_CLASS     ( StringObject, Object );


  explicit inline         StringObject

    ( const String&         s = String() );

  virtual void            readFrom

    ( io::ObjectInput&      in )               override;

  virtual void            writeTo

    ( io::ObjectOutput&     out )        const override;

  virtual String          toString    () const override;
  virtual lint            hashValue   () const override;

  virtual bool            equals

    ( const Ref<Object>&    rhs )        const override;

  virtual Ref<Object>     clone       () const override;

  static Self*            dynamicCast

    ( Object&               rhs );


 public:

  String                  value;


 protected:

  virtual                ~StringObject ();

};







//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class StringObject
//=======================================================================


inline StringObject::StringObject ( const String& s ) : value ( s )
{}


//=======================================================================
//   class DynRefCaster<StringObject,Object>
//=======================================================================


template <>

  class DynRefCaster<StringObject,Object>

{
 public:

  static StringObject*    dynamicCast

    ( const Ref<Object>&    rhs )

  {
    if ( rhs.getType() == Ref<Object>::STRING_TYPE ||
         rhs.getType() == Ref<Object>::OBJECT_TYPE )
    {
      return StringObject::dynamicCast ( *rhs );
    }
    else
    {
      return (StringObject*) nullptr;
    }
  }

};


JEM_END_PACKAGE_BASE

#endif
