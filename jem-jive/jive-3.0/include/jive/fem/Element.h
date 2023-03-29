
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

#ifndef JIVE_FEM_ELEMENT_H
#define JIVE_FEM_ELEMENT_H

#include <jive/geom/Shape.h>
#include <jive/fem/ElementSet.h>
#include <jive/fem/VirtualElement.h>


JIVE_BEGIN_PACKAGE( fem )


//-----------------------------------------------------------------------
//   class Element
//-----------------------------------------------------------------------


class Element : public Object,
                public VirtualElement
{
 public:

  JEM_DECLARE_CLASS       ( Element, Object );

  typedef geom::Shape       Shape;

  enum                      PointType
  {
                              IPOINTS,
                              VERTICES,
                              CUSTOM
  };


  explicit                  Element

    ( const ElementSet&       elems,
      const Ref<Shape>&       shape = nullptr );

  virtual void              reset             () override;

  virtual void              setShape

    ( Shape*                  shape   )          override;

  virtual void              loadData

    ( idx_t                   ielem   )          override;

  inline  void              loadDataFast

    ( idx_t                   ielem   );

  inline  void              selectPoints

    ( PointType               pntType );

  void                      setLocalPoints

    ( const Matrix&           coords  );

  void                      setGlobalPoints

    ( const Matrix&           coords,
      double                  eps     );

  const Matrix&             getGlobalPoints   ();
  const Vector&             getPointWeights   ();
  const Matrix&             getShapeFuncs     ();


 public:

  ElementSet                elems;
  NodeSet                   nodes;

  idx_t                     ielem;
  Ref<Shape>                shape;
  bool                      newShape;

  idx_t                     rank;
  idx_t                     localRank;
  idx_t                     pntCount;
  idx_t                     nodeCount;
  idx_t                     funcCount;

  IdxVector                 inodes;
  Matrix                    xnodes;
  Matrix                    upoints;
  Matrix                    xpoints;

  Vector                    pntWeights;
  Matrix                    shapeFuncs;


 protected:

  static const int          NEW_SHAPE_;
  static const int          NEW_POINTS_;


  virtual                  ~Element           ();

  virtual void              setupChanged_

    ( int                     flags );


 protected:

  PointType                 pntType_;


 private:

  void                      shapeRankError_   () const;

  void                      rankChanged_

    ( idx_t                   newRank );


 private:

  bool                      newShape_;

};





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   loadDataFast
//-----------------------------------------------------------------------


inline void Element::loadDataFast ( idx_t ielem )
{
  this->ielem = ielem;

  elems.getElemNodes  ( inodes, ielem  );
  nodes.getSomeCoords ( xnodes, inodes );

  newShape  = newShape_;
  newShape_ = false;
}


//-----------------------------------------------------------------------
//   selectPoints
//-----------------------------------------------------------------------


inline void Element::selectPoints ( PointType type )
{
  pntType_ = type;

  setupChanged_ ( NEW_POINTS_ );
}


JIVE_END_PACKAGE( fem )

#endif
