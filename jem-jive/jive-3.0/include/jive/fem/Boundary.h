
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

#ifndef JIVE_FEM_BOUNDARY_H
#define JIVE_FEM_BOUNDARY_H

#include <jive/geom/BoundaryShape.h>
#include <jive/geom/InternalShape.h>
#include <jive/fem/BoundarySet.h>
#include <jive/fem/VirtualBoundary.h>
#include <jive/fem/InternalElement.h>


JIVE_BEGIN_PACKAGE( fem )


//-----------------------------------------------------------------------
//   class Boundary
//-----------------------------------------------------------------------


class Boundary : public Object,
                 public VirtualBoundary
{
 public:

  JEM_DECLARE_CLASS       ( Boundary, Object );

  typedef geom::IShape      IShape;
  typedef geom::BShape      BShape;

  enum                      PointType
  {
                              IPOINTS,
                              VERTICES,
                              CUSTOM
  };


  explicit                  Boundary

    ( const BoundarySet&      bounds,
      const Ref<IElement>&    elem = nullptr );

  virtual void              reset             () override;

  virtual void              setShape

    ( IShape*                 shape,
      idx_t                   ilocal  )          override;

  virtual void              loadData

    ( idx_t                   ibound,
      idx_t                   ielem   )          override;

  inline  void              loadDataFast

    ( idx_t                   ibound,
      idx_t                   ielem   );

  inline  void              selectPoints

    ( PointType               pntType );

  inline  void              setLocalPoints

    ( const Matrix&           coords  );

  void                      setGlobalPoints

    ( const Matrix&           coords,
      double                  eps     );

  inline IElement*          getElement        () const;
  const Matrix&             getGlobalPoints   ();
  const Vector&             getPointWeights   ();
  const Matrix&             getNormals        ();
  const Matrix&             getShapeFuncs     ();
  const Cubix&              getShapeGrads     ();


 public:

  BoundarySet               bounds;

  idx_t                     ibound;
  idx_t                     ilocal;

  Ref<BShape>               shape;
  bool                      newShape;

  idx_t                     localRank;
  idx_t                     pntCount;
  idx_t                     nodeCount;
  idx_t                     funcCount;

  IdxVector                 inodes;
  Matrix                    xnodes;
  Matrix                    upoints;
  Matrix                    xpoints;

  Vector                    pntWeights;
  Matrix                    normals;
  Matrix                    shapeFuncs;
  Cubix                     shapeGrads;


 protected:

  static const int          NEW_SHAPE_;
  static const int          NEW_POINTS_;


  virtual                  ~Boundary          ();

  virtual void              setupChanged_

    ( int                     flags );


 protected:

  Ref<IElement>             elem_;
  PointType                 pntType_;
  Vector                    point_;


 private:

  bool                      newShape_;

};




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   loadDataFast
//-----------------------------------------------------------------------


inline void Boundary::loadDataFast

  ( idx_t  ibound,
    idx_t  ielem )

{
  IElement&  e = * elem_;

  e.loadData ( ielem );
  e.shape->getBoundaryCoords ( xnodes, ilocal, e.xnodes );

  this->ibound = ibound;

  newShape     = newShape_;
  newShape_    = false;
}


//-----------------------------------------------------------------------
//   selectPoints
//-----------------------------------------------------------------------


inline void Boundary::selectPoints ( PointType type )
{
  pntType_ = type;

  setupChanged_ ( NEW_POINTS_ );
}


//-----------------------------------------------------------------------
//   getElement
//-----------------------------------------------------------------------


inline IElement* Boundary::getElement () const
{
  return elem_.get ();
}



JIVE_END_PACKAGE( fem )

#endif
