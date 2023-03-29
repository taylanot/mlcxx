
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

#ifndef JIVE_GEOM_STDSQUARE_H
#define JIVE_GEOM_STDSQUARE_H

#include <jem/io/Serializable.h>
#include <jive/geom/import.h>
#include <jive/geom/StdShape.h>


JIVE_BEGIN_PACKAGE( geom )


//-----------------------------------------------------------------------
//   class StdSquare
//-----------------------------------------------------------------------


class StdSquare : public StdShape
{
 public:

  typedef StdSquare       Self;
  typedef StdShape        Super;

  static const int        RANK         = 2;
  static const int        VERTEX_COUNT = 4;

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
      idx_t                 n );

  static Matrix           getNewtonCotesScheme

    ( idx_t                 n );

  static Matrix           getNewtonCotesScheme

    ( idx_t                 m,
      idx_t                 n );

  static Matrix           getIntegrationScheme

    ( String&               ischeme );


 protected:

  virtual                ~StdSquare        ();


 private:

  static idx_t            squareRoot_

    ( idx_t                 n );

};


//-----------------------------------------------------------------------
//   class StdSquare4
//-----------------------------------------------------------------------


class StdSquare4 : public StdSquare,
                   public Serializable
{
 public:

  JEM_DECLARE_CLASS     ( StdSquare4, StdSquare );

  static const int        SHAPE_FUNC_COUNT = 4;


                          StdSquare4       ();

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

  virtual                ~StdSquare4       ();

};


//-----------------------------------------------------------------------
//   class StdSquare6
//-----------------------------------------------------------------------


class StdSquare6 : public StdSquare,
                   public Serializable
{
 public:

  JEM_DECLARE_CLASS     ( StdSquare6, StdSquare );

  static const int        VERTEX_COUNT     = 6;
  static const int        SHAPE_FUNC_COUNT = 6;


                          StdSquare6       ();

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

  virtual                ~StdSquare6       ();

};


//-----------------------------------------------------------------------
//   class StdSquare8
//-----------------------------------------------------------------------


class StdSquare8 : public StdSquare,
                   public Serializable
{
 public:

  JEM_DECLARE_CLASS     ( StdSquare8, StdSquare );

  static const int        VERTEX_COUNT     = 8;
  static const int        SHAPE_FUNC_COUNT = 8;


                          StdSquare8       ();

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

  virtual                ~StdSquare8       ();

};


//-----------------------------------------------------------------------
//   class StdSquare9
//-----------------------------------------------------------------------


class StdSquare9 : public StdSquare,
                   public Serializable
{
 public:

  JEM_DECLARE_CLASS     ( StdSquare9, StdSquare );

  static const int        VERTEX_COUNT     = 9;
  static const int        SHAPE_FUNC_COUNT = 9;


                          StdSquare9       ();

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

  virtual                ~StdSquare9       ();

};



//-----------------------------------------------------------------------
//   class StdSquare12
//-----------------------------------------------------------------------


class StdSquare12 : public StdSquare,
                    public Serializable
{
 public:

  JEM_DECLARE_CLASS     ( StdSquare12, StdSquare );

  static const int        VERTEX_COUNT     = 12;
  static const int        SHAPE_FUNC_COUNT = 12;


                          StdSquare12      ();

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

  virtual                ~StdSquare12      ();

};



//-----------------------------------------------------------------------
//   class StdSquare9B
//-----------------------------------------------------------------------


class StdSquare9B : public StdSquare,
                    public Serializable
{
 public:

  JEM_DECLARE_CLASS     ( StdSquare9B, StdSquare );

  static const int        SHAPE_FUNC_COUNT = 9;


                          StdSquare9B      ();

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

  virtual                ~StdSquare9B      ();

};


//-----------------------------------------------------------------------
//   class StdSquare16B
//-----------------------------------------------------------------------


class StdSquare16B : public StdSquare,
                     public Serializable
{
 public:

  JEM_DECLARE_CLASS     ( StdSquare16B, StdSquare );

  static const int        SHAPE_FUNC_COUNT = 16;


                          StdSquare16B     ();

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

  virtual                ~StdSquare16B     ();

};


//-----------------------------------------------------------------------
//   class StdSquare25B
//-----------------------------------------------------------------------


class StdSquare25B : public StdSquare,
                     public Serializable
{
 public:

  JEM_DECLARE_CLASS     ( StdSquare25B, StdSquare );

  static const int        SHAPE_FUNC_COUNT = 25;


                          StdSquare25B    ();

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

  virtual                ~StdSquare25B    ();

};


//-----------------------------------------------------------------------
//   class StdSquare3Bx2
//-----------------------------------------------------------------------


class StdSquare3Bx2 : public StdSquare,
                      public Serializable
{
 public:

  JEM_DECLARE_CLASS     ( StdSquare3Bx2, StdSquare );

  static const int        SHAPE_FUNC_COUNT = 6;


                          StdSquare3Bx2    ();

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

  virtual                ~StdSquare3Bx2    ();

};


//-----------------------------------------------------------------------
//   class StdSquare4Bx2
//-----------------------------------------------------------------------


class StdSquare4Bx2 : public StdSquare,
                      public Serializable
{
 public:

  JEM_DECLARE_CLASS     ( StdSquare4Bx2, StdSquare );

  static const int        SHAPE_FUNC_COUNT = 8;


                          StdSquare4Bx2    ();

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

  virtual                ~StdSquare4Bx2    ();

};


//-----------------------------------------------------------------------
//   class StdSquare4Bx3B
//-----------------------------------------------------------------------


class StdSquare4Bx3B : public StdSquare,
                       public Serializable
{
 public:

  JEM_DECLARE_CLASS     ( StdSquare4Bx3B, StdSquare );

  static const int        SHAPE_FUNC_COUNT = 12;


                          StdSquare4Bx3B   ();

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

  virtual                ~StdSquare4Bx3B   ();

};


//-----------------------------------------------------------------------
//   class StdSquare5Bx2
//-----------------------------------------------------------------------


class StdSquare5Bx2 : public StdSquare,
                      public Serializable
{
 public:

  JEM_DECLARE_CLASS     ( StdSquare5Bx2, StdSquare );

  static const int        SHAPE_FUNC_COUNT = 10;


                          StdSquare5Bx2    ();

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

  virtual                ~StdSquare5Bx2    ();

};


//-----------------------------------------------------------------------
//   class StdSquare5Bx3B
//-----------------------------------------------------------------------


class StdSquare5Bx3B : public StdSquare,
                       public Serializable
{
 public:

  JEM_DECLARE_CLASS     ( StdSquare5Bx3B, StdSquare );

  static const int        SHAPE_FUNC_COUNT = 15;


                          StdSquare5Bx3B   ();

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

  virtual                ~StdSquare5Bx3B   ();

};


//-----------------------------------------------------------------------
//   class StdSquare6Bx3B
//-----------------------------------------------------------------------


class StdSquare6Bx3B : public StdSquare,
                       public Serializable
{
 public:

  JEM_DECLARE_CLASS     ( StdSquare6Bx3B, StdSquare );

  static const int        SHAPE_FUNC_COUNT = 18;


                          StdSquare6Bx3B   ();

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

  virtual                ~StdSquare6Bx3B   ();

};


//-----------------------------------------------------------------------
//   compatibility typedefs
//-----------------------------------------------------------------------


typedef StdSquare4        StdLinearSquare;
typedef StdSquare8        StdQuadraticSquare;
typedef StdSquare9        StdBiQuadraticSquare;


JIVE_END_PACKAGE( geom )

#endif
