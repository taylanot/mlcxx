
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

#ifndef JIVE_GEOM_STDEDGE_H
#define JIVE_GEOM_STDEDGE_H

#include <jive/geom/StdBoundary.h>


JIVE_BEGIN_PACKAGE( geom )


//-----------------------------------------------------------------------
//   class StdEdge
//-----------------------------------------------------------------------


class StdEdge : public StdBoundary
{
 public:

  JEM_DECLARE_CLASS       ( StdEdge, StdBoundary );


                            StdEdge          ();

                            StdEdge

    ( const Matrix&           xnodes,
      const Ref<StdShape>&    sshape );

                            StdEdge

    ( const Ref<Super>&       parent,
      const Matrix&           xnodes );

  virtual void              evalShapeFunctions

    ( const Vector&           f,
      const Vector&           u )               const override;

  virtual void              evalShapeGradients

    ( const Vector&           f,
      const Matrix&           g,
      const Vector&           u )               const override;

  virtual void              evalShapeGradGrads

    ( const Vector&           f,
      const Matrix&           g,
      const Matrix&           h,
      const Vector&           u )               const override;

  virtual void              mapBoundaryPoint

    ( const Vector&           v,
      const Vector&           u )               const override;


 protected:

  virtual                  ~StdEdge          ();

};


JIVE_END_PACKAGE( geom )

#endif
