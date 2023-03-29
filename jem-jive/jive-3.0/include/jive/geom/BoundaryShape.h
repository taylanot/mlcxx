
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

#ifndef JIVE_GEOM_BOUNDARYSHAPE_H
#define JIVE_GEOM_BOUNDARYSHAPE_H

#include <jive/geom/Shape.h>


JIVE_BEGIN_PACKAGE( geom )


//-----------------------------------------------------------------------
//   class BoundaryShape
//-----------------------------------------------------------------------


class BoundaryShape : public Shape
{
 public:

  JEM_DECLARE_CLASS       ( BoundaryShape, Shape );


  explicit                  BoundaryShape

    ( const String&           name = "" );

  virtual String            getContext    () const override;
  virtual idx_t             globalRank    () const override;

  virtual void              evalNormal

    ( const Vector&           q,
      const Vector&           u,
      const Matrix&           c )            const = 0;

  virtual void              getNormals

    ( const Matrix&           q,
      const Vector&           w,
      const Matrix&           c )            const = 0;

  virtual void              getVertexNormals

    ( const Matrix&           q,
      const Matrix&           c )            const = 0;


 protected:

  virtual                  ~BoundaryShape ();

};



//-----------------------------------------------------------------------
//   related types and functions
//-----------------------------------------------------------------------


typedef BoundaryShape       BShape;


JIVE_END_PACKAGE( geom )

#endif
