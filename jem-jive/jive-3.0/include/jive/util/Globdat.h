
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

#ifndef JIVE_UTIL_GLOBDAT_H
#define JIVE_UTIL_GLOBDAT_H

#include <jem/util/Properties.h>
#include <jive/defines.h>


JIVE_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   class Globdat
//-----------------------------------------------------------------------


class Globdat
{
 public:

  static const char*        CONSTRAINTS;
  static const char*        CONSTRAINTS_PARSER;
  static const char*        DATABASES;
  static const char*        DOF_SPACE;
  static const char*        FUNCTIONS;
  static const char*        ITEM_GROUPS;
  static const char*        ITEM_SETS;
  static const char*        RANDOM;
  static const char*        RELATED_DATA;
  static const char*        RUNTIME;
  static const char*        TABLES;
  static const char*        TEMP;
  static const char*        TIME;
  static const char*        TIME_STEP;
  static const char*        OLD_TIME;
  static const char*        OLD_TIME_STEP;
  static const char*        VARIABLES;
  static const char*        VECTORS;


  static void               init

    ( const Properties&       globdat );

  static Properties         dup

    ( const Properties&       globdat );

  static Properties         newInstance

    ( const String&           name );

  static Properties         newTempData   ();

  static Properties         getVariables

    ( const Properties&       globdat );

  static Properties         getVariables

    ( const String&           name,
      const Properties&       globdat );

  static Properties         findVariables

    ( const String&           name,
      const Properties&       globdat );

  static bool               hasVariable

    ( const String&           name,
      const Properties&       globdat );

  static Properties         findDataFor

    ( const Ref<Object>&      context,
      const Properties&       globdat );

  static Properties         getDataFor

    ( const Ref<Object>&      context,
      const Properties&       globdat );

  static void               eraseDataFor

    ( const Ref<Object>&      context,
      const Properties&       globdat );

  template <class T>
    static inline void      storeFor

    ( const String&           name,
      const Ref<T>&           val,
      const Ref<Object>&      context,
      const Properties&       globdat );

  template <class T>
    static inline bool      findFor

    ( Ref<T>&                 val,
      const String&           name,
      const Ref<Object>&      context,
      const Properties&       globdat );

  template <class T>
    static inline bool      extractFor

    ( Ref<T>&                 val,
      const String&           name,
      const Ref<Object>&      context,
      const Properties&       globdat );

  static void               initTime

    ( const Properties&       globdat );

  static void               advanceTime

    ( double                  delta,
      const Properties&       globdat );

  static void               restoreTime

    ( const Properties&       globdat );

  static void               commitTime

    ( const Properties&       globdat );

  static void               initStep

    ( const Properties&       globdat );

  static void               advanceStep

    ( const Properties&       globdat );

  static void               restoreStep

    ( const Properties&       globdat );

  static void               commitStep

    ( const Properties&       globdat );

};




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   storeFor
//-----------------------------------------------------------------------


template <class T>

  inline void Globdat::storeFor

  ( const String&       name,
    const Ref<T>&       val,
    const Ref<Object>&  context,
    const Properties&   globdat )

{
  Properties  reldat ( getDataFor( context, globdat ) );

  reldat.set ( name, val );
}


//-----------------------------------------------------------------------
//   findFor
//-----------------------------------------------------------------------


template <class T>

  inline bool Globdat::findFor

  ( Ref<T>&             val,
    const String&       name,
    const Ref<Object>&  context,
    const Properties&   globdat )

{
  Properties  reldat ( findDataFor( context, globdat ) );

  return reldat.find ( val, name );
}


//-----------------------------------------------------------------------
//   extractFor
//-----------------------------------------------------------------------


template <class T>

  inline bool Globdat::extractFor

  ( Ref<T>&             val,
    const String&       name,
    const Ref<Object>&  context,
    const Properties&   globdat )

{
  Properties  reldat ( findDataFor( context, globdat ) );

  if ( reldat.find( val, name ) )
  {
    reldat.erase ( name );

    return true;
  }
  else
  {
    return false;
  }
}


JIVE_END_PACKAGE( util )

#endif
