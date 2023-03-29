
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

#ifndef JEM_BASE_BOOLEAN_H
#define JEM_BASE_BOOLEAN_H

#include <jem/base/Object.h>
#include <jem/base/Clonable.h>
#include <jem/io/Serializable.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   class Boolean
//-----------------------------------------------------------------------


class Boolean : public Object,
                public Clonable,
                public io::Serializable
{
 public:

  JEM_DECLARE_CLASS     ( Boolean, Object );


  explicit inline         Boolean

    ( bool                  value = false );

  explicit                Boolean

    ( const String&         s );

  virtual void            readFrom

    ( io::ObjectInput&      in )             override;

  virtual void            writeTo

    ( io::ObjectOutput&     out )      const override;

  virtual String          toString  () const override;
  virtual lint            hashValue () const override;

  virtual bool            equals

    ( const Ref<Object>&    obj )      const override;

  virtual Ref<Object>     clone     () const override;

  static bool             parse

    ( const String&         s );

  static bool             parse

    ( bool&                 value,
      const String&         s );

  static idx_t            print

    ( char*                 buf,
      idx_t                 bufsize,
      bool                  value );

  static Self*            dynamicCast

    ( Object&               rhs );


 public:

  bool                    value;


 protected:

  virtual                ~Boolean   ();

};





//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class Boolean
//=======================================================================


inline Boolean::Boolean ( bool val ) : value ( val )
{}


//=======================================================================
//   class DynRefCaster<Boolean,Object>
//=======================================================================


template <>

  class DynRefCaster<Boolean,Object>

{
 public:

  static Boolean*         dynamicCast

    ( const Ref<Object>&    rhs )

  {
    if ( rhs.getType() == Ref<Object>::BOOL_TYPE ||
         rhs.getType() == Ref<Object>::OBJECT_TYPE )
    {
      return Boolean::dynamicCast ( *rhs );
    }
    else
    {
      return (Boolean*) nullptr;
    }
  }

};


JEM_END_PACKAGE_BASE

#endif
