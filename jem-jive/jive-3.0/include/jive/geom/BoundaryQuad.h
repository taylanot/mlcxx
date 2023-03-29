
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

#ifndef JIVE_GEOM_BOUNDARYQUAD_H
#define JIVE_GEOM_BOUNDARYQUAD_H

#include <jive/geom/typedefs.h>
#include <jive/geom/BoundaryShape.h>


JIVE_BEGIN_PACKAGE( geom )


//-----------------------------------------------------------------------
//   class BoundaryQuad4
//-----------------------------------------------------------------------


class BoundaryQuad4
{
 public:

  static const char*      TYPE_NAME;
  static const char*      ISCHEME;


  static Ref<BShape>      getShape

    ( const String&         name    = "bquad4",
      const Ref<SShape>&    sfuncs  = nullptr );

  static Ref<BShape>      getShape

    ( const String&         name,
      String&               ischeme,
      const Ref<SShape>&    sfuncs  = nullptr );

  static Ref<BShape>      getShape

    ( const String&         name,
      const Matrix&         ischeme,
      const Ref<SShape>&    sfuncs  = nullptr );

  static Ref<BShape>      getShape

    ( const String&         name,
      const Properties&     conf,
      const Properties&     props );

  static void             declare ();

};


//-----------------------------------------------------------------------
//   class BoundaryQuad6
//-----------------------------------------------------------------------


class BoundaryQuad6
{
 public:

  static const char*      TYPE_NAME;
  static const char*      ISCHEME;


  static Ref<BShape>      getShape

    ( const String&         name    = "bquad6",
      const Ref<SShape>&    sfuncs  = nullptr );

  static Ref<BShape>      getShape

    ( const String&         name,
      String&               ischeme,
      const Ref<SShape>&    sfuncs  = nullptr );

  static Ref<BShape>      getShape

    ( const String&         name,
      const Matrix&         ischeme,
      const Ref<SShape>&    sfuncs  = nullptr );

  static Ref<BShape>      getShape

    ( const String&         name,
      const Properties&     conf,
      const Properties&     props );

  static void             declare ();

};


//-----------------------------------------------------------------------
//   class BoundaryQuad8
//-----------------------------------------------------------------------


class BoundaryQuad8
{
 public:

  static const char*      TYPE_NAME;
  static const char*      ISCHEME;


  static Ref<BShape>      getShape

    ( const String&         name    = "bquad8",
      const Ref<SShape>&    sfuncs  = nullptr );

  static Ref<BShape>      getShape

    ( const String&         name,
      String&               ischeme,
      const Ref<SShape>&    sfuncs  = nullptr );

  static Ref<BShape>      getShape

    ( const String&         name,
      const Matrix&         ischeme,
      const Ref<SShape>&    sfuncs  = nullptr );

  static Ref<BShape>      getShape

    ( const String&         name,
      const Properties&     conf,
      const Properties&     props );

  static void             declare ();

};


//-----------------------------------------------------------------------
//   class BoundaryQuad9
//-----------------------------------------------------------------------


class BoundaryQuad9
{
 public:

  static const char*      TYPE_NAME;
  static const char*      ISCHEME;


  static Ref<BShape>      getShape

    ( const String&         name    = "bquad9",
      const Ref<SShape>&    sfuncs  = nullptr );

  static Ref<BShape>      getShape

    ( const String&         name,
      String&               ischeme,
      const Ref<SShape>&    sfuncs  = nullptr );

  static Ref<BShape>      getShape

    ( const String&         name,
      const Matrix&         ischeme,
      const Ref<SShape>&    sfuncs  = nullptr );

  static Ref<BShape>      getShape

    ( const String&         name,
      const Properties&     conf,
      const Properties&     props );

  static void             declare ();

};


//-----------------------------------------------------------------------
//   class BoundaryQuad12
//-----------------------------------------------------------------------


class BoundaryQuad12
{
 public:

  static const char*      TYPE_NAME;
  static const char*      ISCHEME;


  static Ref<BShape>      getShape

    ( const String&         name    = "bquad12",
      const Ref<SShape>&    sfuncs  = nullptr );

  static Ref<BShape>      getShape

    ( const String&         name,
      String&               ischeme,
      const Ref<SShape>&    sfuncs  = nullptr );

  static Ref<BShape>      getShape

    ( const String&         name,
      const Matrix&         ischeme,
      const Ref<SShape>&    sfuncs  = nullptr );

  static Ref<BShape>      getShape

    ( const String&         name,
      const Properties&     conf,
      const Properties&     props );

  static void             declare ();

};


//-----------------------------------------------------------------------
//   compatibility typedefs
//-----------------------------------------------------------------------


typedef BoundaryQuad4     LinearBoundaryQuad;
typedef BoundaryQuad8     QuadraticBoundaryQuad;
typedef BoundaryQuad9     BiQuadraticBoundaryQuad;


JIVE_END_PACKAGE( geom )

#endif
