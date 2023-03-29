
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

#ifndef JIVE_FEM_AUTOSHAPETABLE3D_H
#define JIVE_FEM_AUTOSHAPETABLE3D_H

#include <jive/fem/AutoShapeTable.h>


JIVE_BEGIN_PACKAGE( fem )


class ElementSet;


//-----------------------------------------------------------------------
//   class ShapeTable3D
//-----------------------------------------------------------------------


class AutoShapeTable3D : public AutoShapeTable
{
 public:

  JEM_DECLARE_CLASS       ( AutoShapeTable3D, AutoShapeTable );

  static const char*        TYPE_NAME;


                            AutoShapeTable3D  ();

                            AutoShapeTable3D

    ( const String&           name,
      const ElementSet&       elems,
      idx_t                   maxPrec = 0 );

                            AutoShapeTable3D

    ( const String&           name,
      const ElementGroup&     belems,
      const ElementGroup&     felems,
      idx_t                   maxPrec = 0 );

  static void               initTetProps

    ( const Properties&       props,
      idx_t                   prec,
      idx_t                   nodeCount );

  static void               initHexProps

    ( const Properties&       props,
      idx_t                   prec,
      idx_t                   nodeCount );

  static void               initWedgeProps

    ( const Properties&       props,
      idx_t                   prec,
      idx_t                   nodeCount );

  static void               initBTriangProps

    ( const Properties&       props,
      idx_t                   prec,
      idx_t                   nodeCount );

  static void               initBQuadProps

    ( const Properties&       props,
      idx_t                   prec,
      idx_t                   nodeCount );

  static void               declare           ();

  static Ref<ShapeTable>    makeNew

    ( const String&           name,
      const Properties&       conf,
      const Properties&       props,
      const Ref<ItemSet>&     items,
      const Properties&       globdat );


 protected:

  virtual                  ~AutoShapeTable3D  ();

  virtual bool              makeIShapes_

    ( PrintWriter&            log,
      const ShapeVector&      shapes,
      idx_t                   nodeCount )        override;

  virtual bool              makeBShapes_

    ( PrintWriter&            log,
      const ShapeVector&      shapes,
      idx_t                   nodeCount )        override;

};


JIVE_END_PACKAGE( fem )

#endif
