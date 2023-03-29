
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jive, an object oriented toolkit for solving
 *  partial differential equations.
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
 *  This file is part of Jive, an object oriented toolkit for
 *  solving partial differential equations.
 *
 *  Jive version: 3.0
 *  Date:         Fri 20 Dec 14:30:12 CET 2019
 */

#ifndef JIVE_UTIL_FUNCUTILS_H
#define JIVE_UTIL_FUNCUTILS_H

#include <jem/numeric/func/Function.h>
#include <jive/Array.h>
#include <jive/util/import.h>


JIVE_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   class FuncUtils
//-----------------------------------------------------------------------


class FuncUtils
{
 public:

  class                     NameSpace;
  class                     DeclareNS;

  typedef jem::Array
    < Ref<Function> >       FuncArray;

  static const char*        STD_ARGS;


  static Ref<Function>      newFunc

    ( double                  value = 0.0 );

  static Ref<Function>      newFunc

    ( const String&           expr );

  static Ref<Function>      newFunc

    ( const String&           args,
      const String&           expr );

  static Ref<Function>      newFunc

    ( const String&           expr,
      const Properties&       globdat );

  static Ref<Function>      newFunc

    ( const String&           args,
      const String&           expr,
      const Properties&       globdat );

  static
    inline Ref<Function>    newFunc

    ( const String&           name,
      const Properties&       props,
      const Properties&       globdat );

  static Ref<Function>      newFunc

    ( const String&           args,
      const String&           name,
      const Properties&       props,
      const Properties&       globdat );

  static inline bool        configFunc

    ( Ref<Function>&          func,
      const String&           name,
      const Properties&       props,
      const Properties&       globdat );

  static bool               configFunc

    ( Ref<Function>&          func,
      const String&           args,
      const String&           name,
      const Properties&       props,
      const Properties&       globdat );

  static inline bool        configFuncs

    ( FuncArray&              funcs,
      const String&           name,
      const Properties&       props,
      const Properties&       globdat );

  static bool               configFuncs

    ( FuncArray&              funcs,
      const String&           args,
      const String&           name,
      const Properties&       props,
      const Properties&       globdat );

  static void               getConfig

    ( const Properties&       conf,
      const Ref<Function>&    func,
      const String&           name );

  static void               getConfig

    ( const Properties&       conf,
      const FuncArray&        func,
      const String&           name );

  static double             evalExpr

    ( const String&           expr,
      const Properties&       globdat );

  static double             evalFunc

    ( Function&               func,
      const Properties&       globdat );

  static double             evalFunc

    ( Function&               func,
      const double*           args,
      const Properties&       globdat );

  static Ref<Function>      newCond

    ( bool                    value = true );

  static Ref<Function>      newCond

    ( const char*             expr );

  static Ref<Function>      newCond

    ( const String&           expr );

  static Ref<Function>      newCond

    ( const String&           expr,
      const Properties&       globdat );

  static bool               configCond

    ( Ref<Function>&          cond,
      const String&           name,
      const Properties&       props,
      const Properties&       globdat );

  static bool               evalCond

    ( Function&               cond,
      const Properties&       globdat );

  static void               resolve

    ( Function&               func,
      const Properties&       globdat );

  static void               resolve

    ( const FuncArray&        funcs,
      const Properties&       globdat );

  static void               declareVars

    ( Function&               func,
      const Properties&       globdat );

  static Ref<Function>      getFunc

    ( const String&           name,
      const Properties&       globdat,
      const String&           context );

  static Ref<Function>      findFunc

    ( const String&           name,
      const Properties&       globdat );

  static void               storeFunc

    ( const String&           name,
      const Ref<Function>&    func,
      const Properties&       globdat );

  static Ref<Dictionary>    getFuncs

    ( const Properties&       globdat );

  static Ref<Dictionary>    findFuncs

    ( const Properties&       globdat );

};




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   newFunc
//-----------------------------------------------------------------------


inline Ref<Function> FuncUtils::newFunc

  ( const String&      name,
    const Properties&  props,
    const Properties&  globdat )

{
  return newFunc ( STD_ARGS, name, props, globdat );
}


//-----------------------------------------------------------------------
//   configFunc
//-----------------------------------------------------------------------


inline bool FuncUtils::configFunc

  ( Ref<Function>&     func,
    const String&      name,
    const Properties&  props,
    const Properties&  globdat )

{
  return configFunc ( func, STD_ARGS, name, props, globdat );
}


//-----------------------------------------------------------------------
//   configFuncs
//-----------------------------------------------------------------------


inline bool FuncUtils::configFuncs

  ( FuncArray&         funcs,
    const String&      name,
    const Properties&  props,
    const Properties&  globdat )

{
  return configFuncs ( funcs, STD_ARGS, name, props, globdat );
}


JIVE_END_PACKAGE( util )

#endif
