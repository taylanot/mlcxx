
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

#ifndef JIVE_GEOM_STDLINE_H
#define JIVE_GEOM_STDLINE_H

#include <jem/io/Serializable.h>
#include <jive/geom/import.h>
#include <jive/geom/StdShape.h>


JIVE_BEGIN_PACKAGE( geom )


//-----------------------------------------------------------------------
//   class StdLine
//-----------------------------------------------------------------------


class StdLine : public StdShape
{
 public:

  typedef StdLine         Self;
  typedef StdShape        Super;

  static const int        RANK         = 1;
  static const int        VERTEX_COUNT = 2;

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

    ( idx_t                 count );

  static Matrix           getUniformScheme

    ( idx_t                 count );

  static Matrix           getTrapezoidalScheme

    ( idx_t                 count );

  static Matrix           getNewtonCotesScheme

    ( idx_t                 count );

  static Matrix           getIntegrationScheme

    ( String&               ischeme );

  static idx_t            getGaussPointCount

    ( idx_t                 count );

  static idx_t            getUniformPointCount

    ( idx_t                 count );

  static idx_t            getTrapezoidalPointCount

    ( idx_t                 count );

  static idx_t            getNewtonCotesPointCount

    ( idx_t                 count );


 protected:

  virtual                ~StdLine          ();

};


//-----------------------------------------------------------------------
//   class StdLine2
//-----------------------------------------------------------------------


class StdLine2 : public StdLine,
                 public Serializable
{
 public:

  JEM_DECLARE_CLASS     ( StdLine2, StdLine );

  static const int        SHAPE_FUNC_COUNT = 2;


                          StdLine2         ();

  virtual void            readFrom

    ( ObjectInput&          in )                    override;

  virtual void            writeTo

    ( ObjectOutput&         out )             const override;

  virtual idx_t           shapeFuncCount   () const override;
  virtual bool            isLinear         () const override;

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

  virtual                ~StdLine2         ();

};


//-----------------------------------------------------------------------
//   class StdLine2C1
//-----------------------------------------------------------------------


class StdLine2C1 : public StdLine,
                   public Serializable
{
 public:

  JEM_DECLARE_CLASS     ( StdLine2C1, StdLine );

  static const int        SHAPE_FUNC_COUNT = 4;


                          StdLine2C1       ();

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

  virtual                ~StdLine2C1       ();

};


//-----------------------------------------------------------------------
//   class StdLine3
//-----------------------------------------------------------------------


class StdLine3 : public StdLine,
                 public Serializable
{
 public:

  JEM_DECLARE_CLASS     ( StdLine3, StdLine );

  static const int        VERTEX_COUNT     = 3;
  static const int        SHAPE_FUNC_COUNT = 3;


                          StdLine3         ();

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

  virtual                ~StdLine3         ();

};


//-----------------------------------------------------------------------
//   class StdLine4
//-----------------------------------------------------------------------


class StdLine4 : public StdLine,
                 public Serializable
{
 public:

  JEM_DECLARE_CLASS     ( StdLine4, StdLine );

  static const int        VERTEX_COUNT     = 4;
  static const int        SHAPE_FUNC_COUNT = 4;


                          StdLine4         ();

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

  virtual                ~StdLine4         ();

};


//-----------------------------------------------------------------------
//   class StdLine3B
//-----------------------------------------------------------------------


class StdLine3B : public StdLine,
                  public Serializable
{
 public:

  JEM_DECLARE_CLASS     ( StdLine3B, StdLine );

  static const int        SHAPE_FUNC_COUNT = 3;


                          StdLine3B        ();

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

  virtual                ~StdLine3B        ();

};


//-----------------------------------------------------------------------
//   class StdLine4B
//-----------------------------------------------------------------------


class StdLine4B : public StdLine,
                  public Serializable
{
 public:

  JEM_DECLARE_CLASS     ( StdLine4B, StdLine );

  static const int        SHAPE_FUNC_COUNT = 4;


                          StdLine4B        ();

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

  virtual                ~StdLine4B        ();

};


//-----------------------------------------------------------------------
//   class StdLine5B
//-----------------------------------------------------------------------


class StdLine5B : public StdLine,
                  public Serializable
{
 public:

  JEM_DECLARE_CLASS     ( StdLine5B, StdLine );

  static const int        SHAPE_FUNC_COUNT = 5;


                          StdLine5B        ();

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

  virtual                ~StdLine5B        ();

};


//-----------------------------------------------------------------------
//   class StdLine6B
//-----------------------------------------------------------------------


class StdLine6B : public StdLine,
                  public Serializable
{
 public:

  JEM_DECLARE_CLASS     ( StdLine6B, StdLine );

  static const int        SHAPE_FUNC_COUNT = 6;


                          StdLine6B        ();

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

  virtual                ~StdLine6B        ();

};


//-----------------------------------------------------------------------
//   compatibility typedefs
//-----------------------------------------------------------------------


typedef StdLine2          StdLinearLine;
typedef StdLine3          StdQuadraticLine;


JIVE_END_PACKAGE( geom )

#endif
