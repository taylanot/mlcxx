
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

#ifndef JIVE_GEOM_STDCUBE_H
#define JIVE_GEOM_STDCUBE_H

#include <jem/io/Serializable.h>
#include <jive/geom/import.h>
#include <jive/geom/StdShape.h>


JIVE_BEGIN_PACKAGE( geom )


//-----------------------------------------------------------------------
//   class StdCube
//-----------------------------------------------------------------------


class StdCube : public StdShape
{
 public:

  typedef StdCube         Self;
  typedef StdShape        Super;

  static const int        RANK         = 3;
  static const int        VERTEX_COUNT = 8;

  static const char*      GEOMETRY;


  virtual idx_t           rank             () const override;
  virtual String          getGeometry      () const override;
  virtual idx_t           vertexCount      () const override;
  static  Matrix          vertexCoords     ();
  virtual Matrix          getVertexCoords  () const override;

  virtual bool            containsPoint

    ( const Vector&         u )               const override;

  virtual bool            clipPoint

    ( const Vector&         u )               const override;

  static Matrix           getGaussScheme

    ( idx_t                 n );

  static Matrix           getGaussScheme

    ( idx_t                 m,
      idx_t                 n,
      idx_t                 p );

  static Matrix           getNewtonCotesScheme

    ( idx_t                 n );

  static Matrix           getNewtonCotesScheme

    ( idx_t                 m,
      idx_t                 n,
      idx_t                 p );

  static Matrix           getIntegrationScheme

    ( String&               ischeme );


 protected:

  virtual                ~StdCube          ();


 private:

  static idx_t            cubicRoot_

    ( idx_t                 n );

};


//-----------------------------------------------------------------------
//   class StdCube8
//-----------------------------------------------------------------------


class StdCube8 : public StdCube,
                 public Serializable
{
 public:

  JEM_DECLARE_CLASS     ( StdCube8, StdCube );

  static const int        SHAPE_FUNC_COUNT = 8;


                          StdCube8         ();

  virtual void            readFrom

    ( ObjectInput&          in )                    override;

  virtual void            writeTo

    ( ObjectOutput&         out )             const override;

  virtual idx_t           shapeFuncCount   () const override;

  virtual void            evalShapeFunctions

    ( const Vector&         f,
      const Vector&         u )               const override;

  virtual void            evalShapeGradients

    ( const Vector&         f,
      const Matrix&         g,
      const Vector&         u )               const override;

  virtual void            evalShapeGradGrads

    ( const Vector&         f,
      const Matrix&         g,
      const Matrix&         h,
      const Vector&         u )               const override;

  static Ref<SShape>      makeNew

    ( const String&         geom,
      const Properties&     conf,
      const Properties&     props );

  static void             declare          ();


 protected:

  virtual                ~StdCube8         ();

};


//-----------------------------------------------------------------------
//   class StdCube16
//-----------------------------------------------------------------------


class StdCube16 : public StdCube,
                  public Serializable
{
 public:

  JEM_DECLARE_CLASS     ( StdCube16, StdCube );

  static const int        VERTEX_COUNT     = 16;
  static const int        SHAPE_FUNC_COUNT = 16;


                          StdCube16        ();

  virtual void            readFrom

    ( ObjectInput&          in )                    override;

  virtual void            writeTo

    ( ObjectOutput&         out )             const override;

  virtual idx_t           shapeFuncCount   () const override;
  virtual idx_t           vertexCount      () const override;
  static  Matrix          vertexCoords     ();
  virtual Matrix          getVertexCoords  () const override;

  virtual void            evalShapeFunctions

    ( const Vector&         f,
      const Vector&         u )               const override;

  virtual void            evalShapeGradients

    ( const Vector&         f,
      const Matrix&         g,
      const Vector&         u )               const override;

  virtual void            evalShapeGradGrads

    ( const Vector&         f,
      const Matrix&         g,
      const Matrix&         h,
      const Vector&         u )               const override;

  static Ref<SShape>      makeNew

    ( const String&         geom,
      const Properties&     conf,
      const Properties&     props );

  static void             declare          ();


 protected:

  virtual                ~StdCube16        ();

};


//-----------------------------------------------------------------------
//   class StdCube20
//-----------------------------------------------------------------------


class StdCube20 : public StdCube,
                  public Serializable
{
 public:

  JEM_DECLARE_CLASS     ( StdCube20, StdCube );

  static const int        VERTEX_COUNT     = 20;
  static const int        SHAPE_FUNC_COUNT = 20;


                          StdCube20        ();

  virtual void            readFrom

    ( ObjectInput&          in )                    override;

  virtual void            writeTo

    ( ObjectOutput&         out )             const override;

  virtual idx_t           shapeFuncCount   () const override;
  virtual idx_t           vertexCount      () const override;
  static  Matrix          vertexCoords     ();
  virtual Matrix          getVertexCoords  () const override;

  virtual void            evalShapeFunctions

    ( const Vector&         f,
      const Vector&         u )               const override;

  virtual void            evalShapeGradients

    ( const Vector&         f,
      const Matrix&         g,
      const Vector&         u )               const override;

  virtual void            evalShapeGradGrads

    ( const Vector&         f,
      const Matrix&         g,
      const Matrix&         h,
      const Vector&         u )               const override;

  static Ref<SShape>      makeNew

    ( const String&         geom,
      const Properties&     conf,
      const Properties&     props );

  static void             declare          ();


 protected:

  virtual                ~StdCube20        ();

};


//-----------------------------------------------------------------------
//   class StdCube27B
//-----------------------------------------------------------------------


class StdCube27B : public StdCube,
                   public Serializable
{
 public:

  JEM_DECLARE_CLASS     ( StdCube27B, StdCube );

  static const int        SHAPE_FUNC_COUNT = 27;


                          StdCube27B       ();

  virtual void            readFrom

    ( ObjectInput&          in )                    override;

  virtual void            writeTo

    ( ObjectOutput&         out )             const override;

  virtual idx_t           shapeFuncCount   () const override;

  virtual void            evalShapeFunctions

    ( const Vector&         f,
      const Vector&         u )               const override;

  virtual void            evalShapeGradients

    ( const Vector&         f,
      const Matrix&         g,
      const Vector&         u )               const override;

  virtual void            evalShapeGradGrads

    ( const Vector&         f,
      const Matrix&         g,
      const Matrix&         h,
      const Vector&         u )               const override;

  static Ref<SShape>      makeNew

    ( const String&         geom,
      const Properties&     conf,
      const Properties&     props );

  static void             declare          ();


 protected:

  virtual                ~StdCube27B       ();

};


//-----------------------------------------------------------------------
//   class StdCube3Bx2x2
//-----------------------------------------------------------------------


class StdCube3Bx2x2 : public StdCube,
                      public Serializable
{
 public:

  JEM_DECLARE_CLASS     ( StdCube3Bx2x2, StdCube );

  static const int        SHAPE_FUNC_COUNT = 12;


                          StdCube3Bx2x2    ();

  virtual void            readFrom

    ( ObjectInput&          in )                    override;

  virtual void            writeTo

    ( ObjectOutput&         out )             const override;

  virtual idx_t           shapeFuncCount   () const override;

  virtual void            evalShapeFunctions

    ( const Vector&         f,
      const Vector&         u )               const override;

  virtual void            evalShapeGradients

    ( const Vector&         f,
      const Matrix&         g,
      const Vector&         u )               const override;

  virtual void            evalShapeGradGrads

    ( const Vector&         f,
      const Matrix&         g,
      const Matrix&         h,
      const Vector&         u )               const override;

  static Ref<SShape>      makeNew

    ( const String&         geom,
      const Properties&     conf,
      const Properties&     props );

  static void             declare          ();


 protected:

  virtual                ~StdCube3Bx2x2    ();

};


//-----------------------------------------------------------------------
//   compatibility typedefs
//-----------------------------------------------------------------------


typedef StdCube8          StdLinearCube;
typedef StdCube20         StdQuadraticCube;


JIVE_END_PACKAGE( geom )

#endif
