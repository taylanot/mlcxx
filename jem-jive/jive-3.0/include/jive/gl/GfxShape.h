
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

#ifndef JIVE_GL_GFXSHAPE_H
#define JIVE_GL_GFXSHAPE_H

#include <jem/base/Object.h>
#include <jive/geom/BoundaryShape.h>
#include <jive/Array.h>
#include <jive/gl/import.h>


JIVE_BEGIN_PACKAGE( gl )


class GfxPoint;
class GfxLine;
class GfxFace;


//-----------------------------------------------------------------------
//   class GfxShape
//-----------------------------------------------------------------------


class GfxShape : public Object
{
 public:

  JEM_DECLARE_CLASS       ( GfxShape, Object );

  enum                      TypeID
  {
                              NONE,
                              POINT,
                              LINE,
                              FACE
  };


                            GfxShape      ();

  inline bool               isPoint       () const;
  inline bool               isLine        () const;
  inline bool               isFace        () const;

  inline       GfxPoint*    toPoint       ();
  inline const GfxPoint*    toPoint       () const;

  inline       GfxLine*     toLine        ();
  inline const GfxLine*     toLine        () const;

  inline       GfxFace*     toFace        ();
  inline const GfxFace*     toFace        () const;


 public:

  const TypeID              type;


 protected:

  explicit                  GfxShape

    ( TypeID                  tid );

  virtual                  ~GfxShape      ();

};


//-----------------------------------------------------------------------
//   class GfxPoint
//-----------------------------------------------------------------------


class GfxPoint : public GfxShape
{
 public:

  typedef GfxPoint          Self;
  typedef GfxShape          Super;


                            GfxPoint      ();


 protected:

  virtual                  ~GfxPoint      ();

};


//-----------------------------------------------------------------------
//   class GfxLine
//-----------------------------------------------------------------------


class GfxLine : public GfxShape
{
 public:

  typedef GfxLine           Self;
  typedef GfxShape          Super;


                            GfxLine       ();

  inline idx_t              nodeCount     () const;


 public:

  IdxVector                 inodes;


 protected:

  virtual                  ~GfxLine       ();

};


//-----------------------------------------------------------------------
//   class GfxFace
//-----------------------------------------------------------------------


class GfxFace : public GfxShape
{
 public:

  typedef GfxFace           Self;
  typedef GfxShape          Super;

  typedef jive::
    geom::BoundaryShape     BShape;


  explicit                  GfxFace

    ( const Ref<BShape>&      shape );

  void                      check       () const;
  inline idx_t              nodeCount   () const;
  inline idx_t              edgeCount   () const;
  inline idx_t              xpointCount () const;
  inline idx_t              vertexCount () const;
  inline idx_t              triangCount () const;

  void                      setXPoint

    ( double                  x,
      double                  y );

  void                      setXPoints

    ( const Matrix&           xp );


 public:

  Ref<BShape>               shape;

  IdxVector                 xedges;
  IdxVector                 enodes;
  Matrix                    xfuncs;
  IdxMatrix                 triangles;


 protected:

  virtual                  ~GfxFace     ();


 private:

  void                      checkFailed_

    ( const char*             expr,
      int                     line )       const;

};


//-----------------------------------------------------------------------
//   related types
//-----------------------------------------------------------------------


typedef jem::Array
  < Ref<GfxShape> >         GfxShapeVector;

typedef jem::Array
  < Ref<GfxLine> >          GfxLineVector;

typedef jem::Array
  < Ref<GfxFace> >          GfxFaceVector;





//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class GfxShape
//=======================================================================

//-----------------------------------------------------------------------
//   isPoint
//-----------------------------------------------------------------------


inline bool GfxShape::isPoint () const
{
  return (type == POINT);
}


//-----------------------------------------------------------------------
//   isLine
//-----------------------------------------------------------------------


inline bool GfxShape::isLine () const
{
  return (type == LINE);
}


//-----------------------------------------------------------------------
//   isFace
//-----------------------------------------------------------------------


inline bool GfxShape::isFace () const
{
  return (type == FACE);
}


//-----------------------------------------------------------------------
//   toPoint
//-----------------------------------------------------------------------


inline       GfxPoint*  GfxShape::toPoint ()
{
  JEM_ASSERT ( type == POINT );

  return static_cast<GfxPoint*> ( this );
}


inline const GfxPoint*  GfxShape::toPoint () const
{
  JEM_ASSERT ( type == POINT );

  return static_cast<const GfxPoint*> ( this );
}


//-----------------------------------------------------------------------
//   toLine
//-----------------------------------------------------------------------


inline       GfxLine*   GfxShape::toLine ()
{
  JEM_ASSERT ( type == LINE );

  return static_cast<GfxLine*> ( this );
}


inline const GfxLine*   GfxShape::toLine () const
{
  JEM_ASSERT ( type == LINE );

  return static_cast<const GfxLine*> ( this );
}


//-----------------------------------------------------------------------
//   toFace
//-----------------------------------------------------------------------


inline       GfxFace*   GfxShape::toFace ()
{
  JEM_ASSERT ( type == FACE );

  return static_cast<GfxFace*> ( this );
}


inline const GfxFace*   GfxShape::toFace () const
{
  JEM_ASSERT ( type == FACE );

  return static_cast<const GfxFace*> ( this );
}


//=======================================================================
//   class GfxLine
//=======================================================================


inline idx_t GfxLine::nodeCount () const
{
  return inodes.size ();
}


//=======================================================================
//   class GfxFace
//=======================================================================

//-----------------------------------------------------------------------
//   nodeCount
//-----------------------------------------------------------------------


inline idx_t GfxFace::nodeCount () const
{
  return shape->nodeCount ();
}


//-----------------------------------------------------------------------
//   edgeCount
//-----------------------------------------------------------------------


inline idx_t GfxFace::edgeCount () const
{
  return (xedges.size() - 1);
}


//-----------------------------------------------------------------------
//   xpointCount
//-----------------------------------------------------------------------


inline idx_t GfxFace::xpointCount () const
{
  return xfuncs.size (1);
}


//-----------------------------------------------------------------------
//   vertexCount
//-----------------------------------------------------------------------


inline idx_t GfxFace::vertexCount () const
{
  return (xfuncs.size(1) + shape->nodeCount());
}


//-----------------------------------------------------------------------
//   triangCount
//-----------------------------------------------------------------------


inline idx_t GfxFace::triangCount () const
{
  return triangles.size (1);
}


JIVE_END_PACKAGE( gl )

#endif
