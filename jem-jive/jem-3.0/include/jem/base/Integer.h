
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

#ifndef JEM_BASE_INTEGER_H
#define JEM_BASE_INTEGER_H

#include <jem/base/Number.h>
#include <jem/base/Clonable.h>
#include <jem/io/Serializable.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   class Integer
//-----------------------------------------------------------------------


class Integer : public Number,
                public Clonable,
                public io::Serializable
{
 public:

  JEM_DECLARE_CLASS     ( Integer, Number );

  static const lint       MIN_VALUE;
  static const lint       MAX_VALUE;


  explicit inline         Integer

    ( lint                  value = 0 );

  explicit                Integer

    ( const String&         str );

  virtual void            readFrom

    ( io::ObjectInput&      in )               override;

  virtual void            writeTo

    ( io::ObjectOutput&     out )        const override;

  virtual String          toString    () const override;
  virtual lint            hashValue   () const override;

  virtual bool            equals

    ( const Ref<Object>&    rhs )        const override;

  virtual Ref<Object>     clone       () const override;

  virtual lint            toInteger   () const override;
  virtual double          toFloat     () const override;
  virtual bool            isFloat     () const override;

  static lint             parse

    ( const String&         str );

  static bool             parse

    ( int&                  value,
      const String&         str );

  static bool             parse

    ( long&                 value,
      const String&         str );

  static bool             parse

    ( lint&                 value,
      const String&         str );

  static bool             parse

    ( idx_t&                value,
      const String&         str );

  static idx_t            print

    ( char*                 buf,
      idx_t                 bufsize,
      lint                  value,
      int                   base = 10 );

  static Self*            dynamicCast

    ( Object&               rhs );


 public:

  lint                    value;


 protected:

  virtual                ~Integer     ();

};




//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class Integer
//=======================================================================


inline Integer::Integer ( lint val ) : value ( val )
{}


//=======================================================================
//   class DynRefCaster<Integer,Object>
//=======================================================================


template <>

  class DynRefCaster<Integer,Object>

{
 public:

  static Integer*         dynamicCast

    ( const Ref<Object>&    rhs )

  {
    if ( rhs.getType() == Ref<Object>::INT_TYPE ||
         rhs.getType() == Ref<Object>::OBJECT_TYPE )
    {
      return Integer::dynamicCast ( *rhs );
    }
    else
    {
      return (Integer*) nullptr;
    }
  }

};


JEM_END_PACKAGE_BASE

#endif
