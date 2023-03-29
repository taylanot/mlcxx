
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

#ifndef JIVE_GEOM_INTEGRATIONSCHEMES_H
#define JIVE_GEOM_INTEGRATIONSCHEMES_H

#include <jive/defines.h>


JIVE_BEGIN_PACKAGE( geom )


//-----------------------------------------------------------------------
//   class IntegrationSchemes
//-----------------------------------------------------------------------


class IntegrationSchemes
{
 public:

  static const char*        GAUSS;
  static const char*        UNIFORM;
  static const char*        TRAPEZOIDAL;
  static const char*        NEWTON_COTES;


 public:

  static void               parse

    ( String&                 name,
      idx_t&                  pcount,
      const String&           ischeme );

  static void               split

    ( String&                 part0,
      String&                 part1,
      const String&           ischeme );

  static void               split

    ( String&                 part0,
      String&                 part1,
      String&                 part2,
      const String&           ischeme );

  static void               split

    ( String&                 part0,
      String&                 part1,
      String&                 part2,
      String&                 part3,
      const String&           ischeme );

  static String             join

    ( const String&           part0,
      const String&           part1 );

  static String             join

    ( const String&           part0,
      const String&           part1,
      const String&           part2 );

  static String             join

    ( const String&           part0,
      const String&           part1,
      const String&           part2,
      const String&           part3 );


 private:

  static void               split_

    ( idx_t                   n,
      String**                parts,
      const String&           ischeme );

};


//-----------------------------------------------------------------------
//   related types
//-----------------------------------------------------------------------


typedef IntegrationSchemes  ISchemes;


JIVE_END_PACKAGE( geom )

#endif
