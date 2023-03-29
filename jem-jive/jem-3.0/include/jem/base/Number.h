
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

#ifndef JEM_BASE_NUMBER_H
#define JEM_BASE_NUMBER_H

#include <jem/base/Object.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   class Number
//-----------------------------------------------------------------------


class Number : public Object
{
 public:

  JEM_DECLARE_CLASS     ( Number, Object );


  virtual double          toFloat         () const = 0;
  virtual lint            toInteger       () const = 0;
  virtual bool            isFloat         () const = 0;
  inline  bool            isInteger       () const;

  static Self*            dynamicCast

    ( Object&               rhs );


 protected:

  inline                  Number          () noexcept;
  virtual inline         ~Number          ();

  virtual void            emitVtableFunc_ ();

};




//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class Number
//=======================================================================


inline Number::Number () noexcept
{}


inline Number::~Number ()
{}


inline bool Number::isInteger () const
{
  return ! isFloat ();
}


//=======================================================================
//   class DynRefCaster<Number,Object>
//=======================================================================


template <>

  class DynRefCaster<Number,Object>

{
 public:

  static Number*          dynamicCast

    ( const Ref<Object>&    rhs )

  {
    if ( rhs.getType() == Ref<Object>::INT_TYPE   ||
         rhs.getType() == Ref<Object>::FLOAT_TYPE ||
         rhs.getType() == Ref<Object>::OBJECT_TYPE )
    {
      return Number::dynamicCast ( *rhs );
    }
    else
    {
      return (Number*) nullptr;
    }
  }

};


JEM_END_PACKAGE_BASE

#endif
