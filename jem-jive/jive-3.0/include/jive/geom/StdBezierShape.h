
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

#ifndef JIVE_GEOM_STDBEZIERSHAPE_H
#define JIVE_GEOM_STDBEZIERSHAPE_H

#include <jem/io/Serializable.h>
#include <jive/geom/import.h>
#include <jive/geom/StdShape.h>
#include <jive/geom/BezierExtension.h>
#include <jive/geom/BezierExtractor.h>


JIVE_BEGIN_PACKAGE( geom )


//-----------------------------------------------------------------------
//   class StdBezierShape
//-----------------------------------------------------------------------


class StdBezierShape : public StdShape,
                       public BezierExtension,
                       public Serializable
{
 public:

  JEM_DECLARE_CLASS       ( StdBezierShape, StdShape );


                            StdBezierShape   ();

  explicit                  StdBezierShape

    ( const Ref<StdShape>&    basis );

                            StdBezierShape

    ( const Ref<StdShape>&    basis,
      idx_t                   sfCount );

                            StdBezierShape

    ( const Ref<StdShape>&    basis,
      const BezierExtractor&  bez );

                            StdBezierShape

    ( const Ref<StdShape>&    basis,
      const BezierExtractor&  bez,
      const Vector&           wgt );

  virtual void              readFrom

    ( ObjectInput&            in )                    override;

  virtual void              writeTo

    ( ObjectOutput&           out )             const override;

  virtual idx_t             rank             () const override;
  virtual idx_t             shapeFuncCount   () const override;
  virtual idx_t             basisFuncCount   () const override;
  virtual String            getGeometry      () const override;
  virtual bool              isLinear         () const override;
  virtual idx_t             vertexCount      () const override;
  virtual Matrix            getVertexCoords  () const override;

  virtual bool              containsPoint

    ( const Vector&           u )               const override;

  virtual bool              clipPoint

    ( const Vector&           u )               const override;

  virtual void              evalShapeFunctions

    ( const Vector&           h,
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

  virtual void              setBezierExtractor

    ( const BezierExtractor&  xop )                   override;

  virtual void              setBezierExtractor

    ( const BezierExtractor&  xop,
      const Vector&           wgt )                   override;

  virtual void*             getExtByID

    ( ExtensionID             extID )           const override;

  static Ref<SShape>        makeNew

    ( const String&           geom,
      const Properties&       conf,
      const Properties&       props );

  static void               declare          ();


 protected:

  virtual                  ~StdBezierShape   ();


 private:

  void                      init_

    ( const Ref<StdShape>&    basis );

  void                      shapeChanged_    ();

  double                    applyWeights_

    ( const Vector&           f )               const;

  void                      applyWeights_

    ( const Vector&           f,
      const Matrix&           g )               const;

  void                      applyWeights_

    ( const Vector&           f,
      const Matrix&           g,
      const Matrix&           h )               const;


 private:

  Ref<StdShape>             basis_;
  Vector                    weights_;
  BezierExtractor           extractor_;
  Vector                    shapeFuncs_;
  Matrix                    shapeGrads_;
  Matrix                    shapeGrads2_;
  Vector                    gradWeights_;
  Vector                    grad2Weights_;

  bool                      isRational_;

};


JIVE_END_PACKAGE( geom )

#endif
