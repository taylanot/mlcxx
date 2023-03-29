
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

#ifndef JEM_BASE_FLOAT_H
#define JEM_BASE_FLOAT_H

#include <jem/base/Number.h>
#include <jem/base/Clonable.h>
#include <jem/io/Serializable.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   class Float
//-----------------------------------------------------------------------


class Float : public Number,
              public Clonable,
              public io::Serializable
{
 public:

  JEM_DECLARE_CLASS     ( Float, Number );

  static const double     MIN_VALUE;
  static const double     MAX_VALUE;
  static const double     EPSILON;


  explicit inline         Float

    ( double                value = 0.0 );

  explicit                Float

    ( const String&         s );

  virtual void            readFrom

    ( io::ObjectInput&      in )               override;

  virtual void            writeTo

    ( io::ObjectOutput&     out )        const override;

  virtual String          toString    () const override;
  virtual lint            hashValue   () const override;

  virtual bool            equals

    ( const Ref<Object>&   rhs )         const override;

  virtual Ref<Object>     clone       () const override;

  virtual lint            toInteger   () const override;
  virtual double          toFloat     () const override;
  virtual bool            isFloat     () const override;

  static double           parse

    ( const String&         str );

  static bool             parse

    ( float&                value,
      const String&         str );

  static bool             parse

    ( double&               value,
      const String&         str );

  static idx_t            print

    ( char*                 buf,
      idx_t                 bufsize,
      double                value );

  static bool             isNaN

    ( float                 val )        noexcept;

  static bool             isNaN

    ( double                val )        noexcept;

  static bool             isTiny

    ( float                 val )        noexcept;

  static bool             isTiny

    ( double                val )        noexcept;

  static Self*            dynamicCast

    ( Object&               rhs );


 public:

  double                  value;


 protected:

  virtual                ~Float       ();

};



//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class Float
//=======================================================================


inline Float::Float( double val ) : value ( val )
{}


//=======================================================================
//   class DynRefCaster<Float,Object>
//=======================================================================


template <>

  class DynRefCaster<Float,Object>

{
 public:

  static Float*           dynamicCast

    ( const Ref<Object>&    rhs )

  {
    if ( rhs.getType() == Ref<Object>::FLOAT_TYPE ||
         rhs.getType() == Ref<Object>::OBJECT_TYPE )
    {
      return Float::dynamicCast ( *rhs );
    }
    else
    {
      return (Float*) nullptr;
    }
  }

};


JEM_END_PACKAGE_BASE

#endif
