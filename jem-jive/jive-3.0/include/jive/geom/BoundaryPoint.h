
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

#ifndef JIVE_GEOM_BOUNDARYPOINT_H
#define JIVE_GEOM_BOUNDARYPOINT_H

#include <jem/io/Serializable.h>
#include <jive/geom/BoundaryShape.h>
#include <jive/geom/import.h>


JIVE_BEGIN_PACKAGE( geom )


class StdShape;


//-----------------------------------------------------------------------
//   class BoundaryPoint
//-----------------------------------------------------------------------


class BoundaryPoint : public BoundaryShape,
                      public Serializable
{
 public:

  JEM_DECLARE_CLASS       ( BoundaryPoint, BoundaryShape );


                            BoundaryPoint         ();

                            BoundaryPoint

    ( const String&           name,
      int                     orient,
      double                  point,
      const Ref<StdShape>&    xfuncs,
      const Ref<StdShape>&    sfuncs  );

  virtual void              readFrom

    ( ObjectInput&            in )                         override;

  virtual void              writeTo

    ( ObjectOutput&           out )                  const override;

  virtual idx_t             localRank             () const override;
  virtual idx_t             nodeCount             () const override;
  virtual idx_t             vertexCount           () const override;
  virtual idx_t             ipointCount           () const override;
  virtual idx_t             shapeFuncCount        () const override;
  virtual String            getGeometry           () const override;
  virtual Matrix            getVertexCoords       () const override;
  virtual Matrix            getIntegrationScheme  () const override;

  virtual void              getGlobalIntegrationPoints

    ( const Matrix&           x,
      const Matrix&           c )                    const override;

  virtual void              getIntegrationWeights

    ( const Vector&           w,
      const Matrix&           c )                    const override;

  virtual void              getGlobalPoint

    ( const Vector&           x,
      const Vector&           u,
      const Matrix&           c )                    const override;

  virtual bool              getLocalPoint

    ( const Vector&           u,
      const Vector&           x,
      double                  eps,
      const Matrix&           c )                    const override;

  virtual bool              containsLocalPoint

    ( const Vector&           u )                    const override;

  virtual Matrix            getShapeFunctions     () const override;
  virtual Matrix            getVertexFunctions    () const override;

  virtual void              evalShapeFunctions

    ( const Vector&           h,
      const Vector&           u )                    const override;

  virtual void              evalNormal

    ( const Vector&           q,
      const Vector&           u,
      const Matrix&           c )                    const override;

  virtual void              getNormals

    ( const Matrix&           q,
      const Vector&           w,
      const Matrix&           c )                    const override;

  virtual void              getVertexNormals

    ( const Matrix&           q,
      const Matrix&           c )                    const override;

  virtual void*             getExtByID

    ( ExtensionID             extID )                const override;


 protected:

  virtual                  ~BoundaryPoint         ();

  void                      init_                 ();
  void                      shapeChanged_         ();
  void                      xshapeChanged_        ();
  void                      sshapeChanged_        ();

  void                      getGlobalPoint_

    ( double&                 x,
      const Matrix&           c )                    const;

  void                      updateXFuncs_         ();
  void                      updateSFuncs_         ();


 private:

  static const int          U_FUNCS_;
  static const int          U_XFUNCS_;
  static const int          U_SFUNCS_;

  Ref<StdShape>             xshape_;
  Ref<StdShape>             sshape_;
  Vector                    point_;
  Vector                    xfuncs_;
  Matrix                    sfuncs_;
  int                       orient_;
  int                       updated_;

};


JIVE_END_PACKAGE( geom )

#endif
