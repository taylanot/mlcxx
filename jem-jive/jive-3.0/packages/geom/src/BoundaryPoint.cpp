
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


#include <jem/base/assert.h>
#include <jem/base/limits.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/array/utilities.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/Event.h>
#include <jive/geom/StdShape.h>
#include <jive/geom/Geometries.h>
#include <jive/geom/BoundaryPoint.h>


JEM_DEFINE_SERIABLE_CLASS( jive::geom::BoundaryPoint );


JIVE_BEGIN_PACKAGE( geom )


//=======================================================================
//   class BoundaryPoint
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int  BoundaryPoint::U_XFUNCS_ = 1 << 0;
const int  BoundaryPoint::U_SFUNCS_ = 1 << 1;
const int  BoundaryPoint::U_FUNCS_  = U_XFUNCS_ | U_SFUNCS_;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


BoundaryPoint::BoundaryPoint ()
{
  orient_  = 0;
  updated_ = 0;
}


BoundaryPoint::BoundaryPoint

  ( const String&         name,
    int                   orient,
    double                point,
    const Ref<StdShape>&  xfuncs,
    const Ref<StdShape>&  sfuncs ) :

    Super   (   name ),
    xshape_ ( xfuncs ),
    sshape_ ( sfuncs ),
    point_  (      1 ),
    orient_ ( orient )

{
  JEM_PRECHECK2 ( xfuncs == nullptr || xfuncs->rank() == 1,
                  "invalid geometric shape functions" );
  JEM_PRECHECK2 ( sfuncs != nullptr && sfuncs->rank() == 1,
                  "invalid shape functions" );
  JEM_PRECHECK2 ( orient == -1  || orient         == 1,
                  "invalid orientation" );

  point_[0] = point;

  init_ ();
}


BoundaryPoint::~BoundaryPoint ()
{}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void BoundaryPoint::readFrom ( jem::io::ObjectInput& in )
{
  point_.resize ( 1 );

  decode ( in, myName_, point_[0], orient_, sshape_ );

  init_  ();
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void BoundaryPoint::writeTo ( jem::io::ObjectOutput& out ) const
{
  encode ( out, myName_, point_[0], orient_, sshape_ );
}


//-----------------------------------------------------------------------
//   localRank
//-----------------------------------------------------------------------


idx_t BoundaryPoint::localRank () const
{
  return 0;
}


//-----------------------------------------------------------------------
//   nodeCount
//-----------------------------------------------------------------------


idx_t BoundaryPoint::nodeCount () const
{
  return xfuncs_.size ();
}


//-----------------------------------------------------------------------
//   vertexCount
//-----------------------------------------------------------------------


idx_t BoundaryPoint::vertexCount () const
{
  return 1_idx;
}


//-----------------------------------------------------------------------
//   ipointCount
//-----------------------------------------------------------------------


idx_t BoundaryPoint::ipointCount () const
{
  return 1_idx;
}


//-----------------------------------------------------------------------
//   shapeFuncCount
//-----------------------------------------------------------------------


idx_t BoundaryPoint::shapeFuncCount () const
{
  return sfuncs_.size (0);
}


//-----------------------------------------------------------------------
//   getGeometry
//-----------------------------------------------------------------------


String BoundaryPoint::getGeometry () const
{
  return Geometries::POINT;
}


//-----------------------------------------------------------------------
//   getVertexCoords
//-----------------------------------------------------------------------


Matrix BoundaryPoint::getVertexCoords () const
{
  return Matrix ( 0, 1 );
}


//-----------------------------------------------------------------------
//   getIntegrationScheme
//-----------------------------------------------------------------------


Matrix BoundaryPoint::getIntegrationScheme () const
{
  Matrix  s ( 1, 1 );

  s(0,0) = 1.0;

  return s;
}


//-----------------------------------------------------------------------
//   getGlobalIntegrationPoints
//-----------------------------------------------------------------------


void BoundaryPoint::getGlobalIntegrationPoints

  ( const Matrix&  x,
    const Matrix&  c ) const

{
  getGlobalPoint_ ( x(0,0), c );
}


//-----------------------------------------------------------------------
//   getIntegrationWeights
//-----------------------------------------------------------------------


void BoundaryPoint::getIntegrationWeights

  ( const Vector&  w,
    const Matrix&  c ) const

{
  w[0] = 1.0;
}


//-----------------------------------------------------------------------
//   getGlobalPoint
//-----------------------------------------------------------------------


void BoundaryPoint::getGlobalPoint

  ( const Vector&  x,
    const Vector&  u,
    const Matrix&  c ) const

{
  getGlobalPoint_ ( x[0], c );
}


//-----------------------------------------------------------------------
//   getLocalPoint
//-----------------------------------------------------------------------


bool BoundaryPoint::getLocalPoint

  ( const Vector&  u,
    const Vector&  x,
    double         eps,
    const Matrix&  c ) const

{
  double  d, y;

  getGlobalPoint_ ( y, c );

  d = y - x[0];

  if ( d * d <= eps * eps )
  {
    return true;
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------
//   containsLocalPoint
//-----------------------------------------------------------------------


bool BoundaryPoint::containsLocalPoint

  ( const Vector& u ) const

{
  JEM_ASSERT2 ( u.size() == 0,
                "local coordinate array has wrong size" );

  return true;
}


//-----------------------------------------------------------------------
//   getShapeFunctions
//-----------------------------------------------------------------------


Matrix BoundaryPoint::getShapeFunctions () const
{
  if ( (updated_ & U_SFUNCS_) != U_SFUNCS_ )
  {
    const_cast<Self*>( this )->updateSFuncs_ ();
  }

  return sfuncs_;
}


//-----------------------------------------------------------------------
//   getVertexFunctions
//-----------------------------------------------------------------------


Matrix BoundaryPoint::getVertexFunctions () const
{
  return Self::getShapeFunctions ();
}


//-----------------------------------------------------------------------
//   evalShapeFunctions
//-----------------------------------------------------------------------


void BoundaryPoint::evalShapeFunctions

  ( const Vector&  h,
    const Vector&  u ) const

{
  sshape_->evalShapeFunctions ( h, point_ );
}


//-----------------------------------------------------------------------
//   evalNormal
//-----------------------------------------------------------------------


void BoundaryPoint::evalNormal

  ( const Vector&  q,
    const Vector&  u,
    const Matrix&  c ) const

{
  q[0] = (double) orient_;
}


//-----------------------------------------------------------------------
//   getNormals
//-----------------------------------------------------------------------


void BoundaryPoint::getNormals

  ( const Matrix&  q,
    const Vector&  w,
    const Matrix&  c ) const

{
  q(0,0) = (double) orient_;
  w[0]   = 1.0;
}


//-----------------------------------------------------------------------
//   getVertexNormals
//-----------------------------------------------------------------------


void BoundaryPoint::getVertexNormals

  ( const Matrix&  q,
    const Matrix&  c ) const

{
  q(0,0) = (double) orient_;
}


//-----------------------------------------------------------------------
//   getExtByID
//-----------------------------------------------------------------------


void* BoundaryPoint::getExtByID ( ExtensionID id ) const
{
  void*  ext = nullptr;

  if ( xshape_ )
  {
    ext = xshape_->getExtByID ( id );
  }

  if ( ! ext )
  {
    sshape_->getExtByID ( id );
  }

  return ext;
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void BoundaryPoint::init_ ()
{
  using jem::util::connect;

  updated_ = 0;

  if ( ! xshape_ )
  {
    xfuncs_.resize ( 1 );

    xfuncs_[0] = 1.0;
    updated_  |= U_XFUNCS_;
  }
  else
  {
    xfuncs_.resize ( xshape_->shapeFuncCount() );
  }

  if ( xshape_ == sshape_ )
  {
    sfuncs_.ref ( reshape( xfuncs_, xfuncs_.size(), 1 ) );

    connect ( xshape_->changeEvent, this, &Self::shapeChanged_ );
  }
  else
  {
    sfuncs_.resize ( sshape_->shapeFuncCount(), 1 );

    if ( xshape_ )
    {
      connect ( xshape_->changeEvent, this, &Self::xshapeChanged_ );
    }

    connect ( sshape_->changeEvent, this, &Self::sshapeChanged_ );
  }
}


//-----------------------------------------------------------------------
//   shapeChanged_
//-----------------------------------------------------------------------


void BoundaryPoint::shapeChanged_ ()
{
  updated_ &= ~U_FUNCS_;

  changeEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   xshapeChanged_
//-----------------------------------------------------------------------


void BoundaryPoint::xshapeChanged_ ()
{
  updated_ &= ~U_XFUNCS_;

  changeEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   sshapeChanged_
//-----------------------------------------------------------------------


void BoundaryPoint::sshapeChanged_ ()
{
  updated_ &= ~U_SFUNCS_;

  changeEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   getGlobalPoint_
//-----------------------------------------------------------------------


void BoundaryPoint::getGlobalPoint_

  ( double&        x,
    const Matrix&  c ) const

{
  const idx_t  nodeCount = xfuncs_.size ();

  if ( (updated_ & U_XFUNCS_) != U_XFUNCS_ )
  {
    const_cast<Self*>( this )->updateXFuncs_ ();
  }

  switch ( nodeCount )
  {
  case 1:

    x = xfuncs_[0] * c(0,0);
    break;

  case 2:

    x = xfuncs_[0] * c(0,0) + xfuncs_[1] * c(0,1);
    break;

  case 3:

    x = xfuncs_[0] * c(0,0) + xfuncs_[1] * c(0,1) +
        xfuncs_[2] * c(0,2);
    break;

  default:

    x = 0.0;

    for ( idx_t i = 0; i < nodeCount; i++ )
    {
      x += xfuncs_[i] * c(0,i);
    }
  }
}


//-----------------------------------------------------------------------
//   updateXFuncs_
//-----------------------------------------------------------------------


void BoundaryPoint::updateXFuncs_ ()
{
  xshape_->evalShapeFunctions ( xfuncs_, point_ );

  updated_ |= U_XFUNCS_;

  if ( sshape_ == xshape_ )
  {
    updated_ |= U_SFUNCS_;
  }
}


//-----------------------------------------------------------------------
//   updateSFuncs_
//-----------------------------------------------------------------------


void BoundaryPoint::updateSFuncs_ ()
{
  sshape_->evalShapeFunctions ( sfuncs_[0], point_ );

  updated_ |= U_SFUNCS_;

  if ( sshape_ == xshape_ )
  {
    updated_ |= U_XFUNCS_;
  }
}


JIVE_END_PACKAGE( geom )
