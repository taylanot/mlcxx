
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

#ifndef JEM_NUMERIC_FUNC_FUNCTION_H
#define JEM_NUMERIC_FUNC_FUNCTION_H

#include <jem/base/assert.h>
#include <jem/base/Object.h>


JEM_BEGIN_PACKAGE( numeric )


//-----------------------------------------------------------------------
//   class Function
//-----------------------------------------------------------------------


class Function : public Object
{
 public:

  JEM_DECLARE_CLASS       ( Function, Object );


  virtual idx_t             argCount    () const = 0;

  virtual double            getValue

    ( const double*           args )       const = 0;

  virtual double            getDeriv

    ( idx_t                   iarg,
      const double*           args )       const;

  virtual idx_t             markUsedArgs

    ( bool*                   mask )       const;

  inline  double            eval        () const;

  inline  double            eval

    ( double                  arg )        const;

  inline  double            eval

    ( double                  arg0,
      double                  arg1 )       const;

  inline  double            eval

    ( double                  arg0,
      double                  arg1,
      double                  arg2 )       const;

  inline  double            deriv

    ( double                  arg )        const;

  inline  double            deriv

    ( idx_t                   iarg,
      double                  arg0,
      double                  arg1 )       const;

  inline  double            deriv

    ( idx_t                   iarg,
      double                  arg0,
      double                  arg1,
      double                  arg2 )       const;


 protected:

  virtual                  ~Function    ();

};



//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   eval
//-----------------------------------------------------------------------


inline double Function::eval () const
{
  JEM_ASSERT2 ( argCount() == 0,
                "Function should have zero arguments" );

  return getValue ( 0 );
}


inline double Function::eval ( double arg ) const
{
  JEM_ASSERT2 ( argCount() == 1,
                "Function should have one argument" );

  return getValue ( & arg );
}


inline double Function::eval

  ( double  arg0,
    double  arg1 ) const

{
  JEM_ASSERT2 ( argCount() == 2,
                "Function should have two arguments" );

  double  args[2] = { arg0, arg1 };

  return getValue ( args );
}


inline double Function::eval

  ( double  arg0,
    double  arg1,
    double  arg2 ) const

{
  JEM_ASSERT2 ( argCount() == 3,
                "Function should have three arguments" );

  double  args[3] = { arg0, arg1, arg2 };

  return getValue ( args );
}


//-----------------------------------------------------------------------
//   deriv
//-----------------------------------------------------------------------


inline double Function::deriv ( double arg ) const
{
  JEM_ASSERT2 ( argCount() == 1,
                "Function should have one argument" );

  return getDeriv ( 0, & arg );
}


inline double Function::deriv

  ( idx_t   iarg,
    double  arg0,
    double  arg1 ) const

{
  JEM_ASSERT2 ( argCount() == 2,
                "Function should have two arguments" );
  JEM_ASSERT2 ( iarg >= 0 && iarg < 2,
                "invalid Function argument index" );

  double  args[2] = { arg0, arg1 };

  return getDeriv ( iarg, args );
}


inline double Function::deriv

  ( idx_t   iarg,
    double  arg0,
    double  arg1,
    double  arg2 ) const

{
  JEM_ASSERT2 ( argCount() == 3,
                "Function should have three arguments" );
  JEM_ASSERT2 ( iarg >= 0 && iarg < 3,
                "invalid Function argument index" );

  double  args[3] = { arg0, arg1, arg2 };

  return getDeriv ( iarg, args );
}


JEM_END_PACKAGE( numeric )

#endif
