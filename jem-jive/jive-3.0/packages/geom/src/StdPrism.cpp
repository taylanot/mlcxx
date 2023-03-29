
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


#include <jem/pragmas.h>
#include <jem/base/assert.h>
#include <jem/base/ClassTemplate.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/Event.h>
#include <jive/geom/StdPrism.h>


JEM_DEFINE_SERIABLE_CLASS( jive::geom::StdPrism );


JIVE_BEGIN_PACKAGE( geom )


using jem::checkedCast;


//=======================================================================
//   class StdPrism
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


StdPrism::StdPrism ()
{}


StdPrism::StdPrism

  ( const String&         geom,
    const Ref<StdShape>&  base,
    const Ref<StdShape>&  line ) :

    base_     ( base ),
    line_     ( line ),
    geometry_ ( geom )

{
  JEM_PRECHECK2 ( base,
                  "NULL base shape" );
  JEM_PRECHECK2 ( line && line->rank() == 1,
                  "invalid line shape" );

  init_ ();
}


StdPrism::~StdPrism ()
{}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void StdPrism::readFrom ( jem::io::ObjectInput& in )
{
  decode ( in, base_, line_, geometry_ );

  init_ ();
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void StdPrism::writeTo ( jem::io::ObjectOutput& out ) const
{
  encode ( out, base_, line_, geometry_ );
}


//-----------------------------------------------------------------------
//   rank
//-----------------------------------------------------------------------


idx_t StdPrism::rank () const
{
  return (baseRank_ + 1);
}


//-----------------------------------------------------------------------
//   shapeFuncCount
//-----------------------------------------------------------------------


idx_t StdPrism::shapeFuncCount () const
{
  return (base_->shapeFuncCount() * line_->shapeFuncCount());
}


//-----------------------------------------------------------------------
//   getGeometry
//-----------------------------------------------------------------------


String StdPrism::getGeometry () const
{
  return geometry_;
}


//-----------------------------------------------------------------------
//   vertexCount
//-----------------------------------------------------------------------


idx_t StdPrism::vertexCount () const
{
  return (base_->vertexCount() * line_->vertexCount());
}


//-----------------------------------------------------------------------
//   getVertexCoords
//-----------------------------------------------------------------------


Matrix StdPrism::getVertexCoords () const
{
  return getCoords ( base_->getVertexCoords (),
                     line_->getVertexCoords () );
}


//-----------------------------------------------------------------------
//   containsPoint
//-----------------------------------------------------------------------


bool StdPrism::containsPoint ( const Vector& u ) const
{
  linePoint_[0] = u[baseRank_];

  if ( ! line_->containsPoint( linePoint_ ) )
  {
    return false;
  }
  else
  {
    return base_->containsPoint ( u[slice(BEGIN,baseRank_)] );
  }
}


//-----------------------------------------------------------------------
//   clipPoint
//-----------------------------------------------------------------------


bool StdPrism::clipPoint ( const Vector& u ) const
{
  linePoint_[0] = u[baseRank_];

  if ( line_->clipPoint( linePoint_ ) )
  {
    u[baseRank_] = linePoint_[0];

    base_->clipPoint ( u[slice(BEGIN,baseRank_)] );

    return true;
  }
  else
  {
    return base_->clipPoint ( u[slice(BEGIN,baseRank_)] );
  }
}


//-----------------------------------------------------------------------
//   evalShapeFunctions
//-----------------------------------------------------------------------


void StdPrism::evalShapeFunctions

  ( const Vector&  f,
    const Vector&  u ) const

{
  const Vector&  fb = baseFuncs_;
  const Vector&  fl = lineFuncs_;

  const idx_t  baseFuncCount = baseFuncs_.size ();
  const idx_t  lineFuncCount = lineFuncs_.size ();

  Vector       basePoint ( u[slice(BEGIN,baseRank_)] );


  linePoint_[0] = u[baseRank_];

  base_->evalShapeFunctions ( fb, basePoint );
  line_->evalShapeFunctions ( fl, linePoint_ );

  for ( idx_t i = 0; i < lineFuncCount; i++ )
  {
    double  fli = fl[i];

JEM_IVDEP

    for ( idx_t j = 0; j < baseFuncCount; j++ )
    {
      f[i * baseFuncCount + j] = fli * fb[j];
    }
  }
}


//-----------------------------------------------------------------------
//   evalShapeGradients
//-----------------------------------------------------------------------


void StdPrism::evalShapeGradients

  ( const Vector&  f,
    const Matrix&  g,
    const Vector&  u ) const

{
  const Vector&  fb = baseFuncs_;
  const Vector&  fl = lineFuncs_;
  const Matrix&  gb = baseGrads_;
  const Matrix&  gl = lineGrads_;

  const idx_t  baseRank      = baseRank_;
  const idx_t  baseFuncCount = baseFuncs_.size ();
  const idx_t  lineFuncCount = lineFuncs_.size ();

  Vector       basePoint ( u[slice(BEGIN,baseRank)] );


  linePoint_[0] = u[baseRank];

  base_->evalShapeGradients ( fb, gb, basePoint );
  line_->evalShapeGradients ( fl, gl, linePoint_ );

  for ( idx_t i = 0; i < lineFuncCount; i++ )
  {
    double  fli = fl[i];
    double  gli = gl(0,i);

    for ( idx_t j = 0; j < baseFuncCount; j++ )
    {
      idx_t  ifunc = i * baseFuncCount + j;

      f[ifunc] = fli * fb[j];

JEM_IVDEP

      for ( idx_t ii = 0; ii < baseRank; ii++ )
      {
        g(ii,ifunc) = fli * gb(ii,j);
      }

      g(baseRank,ifunc) = gli * fb[j];
    }
  }
}


//-----------------------------------------------------------------------
//   evalShapeGradGrads
//-----------------------------------------------------------------------


void StdPrism::evalShapeGradGrads

  ( const Vector&  f,
    const Matrix&  g,
    const Matrix&  h,
    const Vector&  u ) const

{
  const Vector&  fb = baseFuncs_;
  const Vector&  fl = lineFuncs_;
  const Matrix&  gb = baseGrads_;
  const Matrix&  gl = lineGrads_;
  const Matrix&  hb = baseGrads2_;
  const Matrix&  hl = lineGrads2_;

  const idx_t  baseRank      = baseRank_;
  const idx_t  baseFuncCount = baseFuncs_.size ();
  const idx_t  lineFuncCount = lineFuncs_.size ();

  Vector       basePoint ( u[slice(BEGIN,baseRank)] );


  linePoint_[0] = u[baseRank];

  base_->evalShapeGradGrads ( fb, gb, hb, basePoint );
  line_->evalShapeGradGrads ( fl, gl, hl, linePoint_ );

  for ( idx_t i = 0; i < lineFuncCount; i++ )
  {
    double  fli = fl[i];
    double  gli = gl(0,i);
    double  hli = hl(0,i);

    for ( idx_t j = 0; j < baseFuncCount; j++ )
    {
      idx_t  ifunc = i * baseFuncCount + j;

      f[ifunc] = fli * fb[j];

JEM_IVDEP

      for ( idx_t ii = 0; ii < baseRank; ii++ )
      {
        g(ii,ifunc) = fli * gb(ii,j);
        h(ii,ifunc) = fli * hb(ii,j);
      }

      g(baseRank,ifunc) = gli * fb[j];
      h(baseRank,ifunc) = hli * fb[j];

      idx_t  ik = baseRank + 1;
      idx_t  jk = baseRank;

      for ( idx_t ii = 0; ii < baseRank; ii++, ik++ )
      {

JEM_IVDEP

        for ( idx_t jj = ii + 1; jj < baseRank; jj++, ik++, jk++ )
        {
          h(ik,ifunc) = fli * hb(jk,j);
        }

        h(ik,ifunc) = gli * gb(ii,j);
      }
    }
  }
}


//-----------------------------------------------------------------------
//   getExtByID
//-----------------------------------------------------------------------


void* StdPrism::getExtByID ( ExtensionID id ) const
{
  void*  ext = base_->getExtByID ( id );

  if ( ! ext )
  {
    ext = line_->getExtByID ( id );
  }

  return ext;
}


//-----------------------------------------------------------------------
//   getCoords
//-----------------------------------------------------------------------


Matrix StdPrism::getCoords

  ( const Matrix&  xb,
    const Matrix&  xl )

{
  JEM_PRECHECK2 ( xl.size(0) == 1,
                  "coordinate array has wrong shape" );

  const idx_t  baseRank       = xb.size(0);
  const idx_t  basePointCount = xb.size(1);
  const idx_t  linePointCount = xl.size(1);

  Matrix  x ( baseRank + 1, basePointCount * linePointCount );


  for ( idx_t i = 0; i < linePointCount; i++ )
  {
    double  a = xl(0,i);

    for ( idx_t j = 0; j < basePointCount; j++ )
    {
      idx_t  k = i * basePointCount + j;

JEM_IVDEP

      for ( idx_t r = 0; r < baseRank; r++ )
      {
        x(r,k) = xb(r,j);
      }

      x(baseRank,k) = a;
    }
  }

  return x;
}


//-----------------------------------------------------------------------
//   getIntegrationScheme
//-----------------------------------------------------------------------


Matrix StdPrism::getIntegrationScheme

  ( const Matrix&  sb,
    const Matrix&  sl )

{
  JEM_PRECHECK2 ( sb.size(0) > 1 && sl.size(0) == 2,
                  "Array shape mismatch" );

  const idx_t  baseRank        = sb.size(0) - 1;
  const idx_t  basePointCount  = sb.size(1);
  const idx_t  linePointCount  = sl.size(1);

  Matrix  s ( baseRank + 2, basePointCount * linePointCount );


  for ( idx_t i = 0; i < linePointCount; i++ )
  {
    double  a = sl(0,i);
    double  b = sl(1,i);

    for ( idx_t j = 0; j < basePointCount; j++ )
    {
      idx_t  k = i * basePointCount + j;
      s(0,k)   = a * sb(0,j);

JEM_IVDEP

      for ( idx_t r = 1; r <= baseRank; r++ )
      {
        s(r,k) = sb(r,j);
      }

      s(baseRank + 1,k) = b;
    }
  }

  return s;
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void StdPrism::init_ ()
{
  const idx_t  baseFuncCount = base_->shapeFuncCount ();
  const idx_t  lineFuncCount = line_->shapeFuncCount ();

  idx_t        baseRank2;


  baseRank_  = base_->rank ();
  baseRank2  = (baseRank_ * baseRank_ + baseRank_) / 2;

  linePoint_ .resize (                        1 );
  baseFuncs_ .resize (            baseFuncCount );
  lineFuncs_ .resize (            lineFuncCount );
  baseGrads_ .resize ( baseRank_, baseFuncCount );
  lineGrads_ .resize ( 1,         lineFuncCount );
  baseGrads2_.resize ( baseRank2, baseFuncCount );
  lineGrads2_.resize ( 1,         lineFuncCount );

  connect ( base_->changeEvent, this, & Self::shapeChanged_ );
  connect ( line_->changeEvent, this, & Self::shapeChanged_ );
}


//-----------------------------------------------------------------------
//   shapeChanged_
//-----------------------------------------------------------------------


void StdPrism::shapeChanged_ ()
{
  changeEvent.emit ( *this );
}


JIVE_END_PACKAGE( geom )
