
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


#include <cmath>
#include <jem/pragmas.h>
#include <jem/base/limits.h>
#include <jem/base/assert.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/array/operators.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/Event.h>
#include <jem/numeric/algebra/matmul.h>
#include <jem/numeric/algebra/LUSolver.h>
#include <jive/geom/error.h>
#include <jive/geom/StdShape.h>
#include <jive/geom/ParametricUtils.h>
#include <jive/geom/ParametricBoundary.h>


JEM_DEFINE_CLASS( jive::geom::ParametricBoundary );


JIVE_BEGIN_PACKAGE( geom )


using jem::checkedCast;
using jem::numeric::matmul;


//=======================================================================
//   class ParametricBoundary
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int     ParametricBoundary::MAX_NR_ITER_ = 20;
const double  ParametricBoundary::NR_ALPHA_    = 1.0e-2;
const double  ParametricBoundary::NR_BETA_     = 1.0e20;
const double  ParametricBoundary::NR_DELTA_    = 1.0e-4;

const int     ParametricBoundary::U_NONE_      = 0;

const int     ParametricBoundary::U_XFUNCS_    = (1 << 0);
const int     ParametricBoundary::U_XGRADS_    = (1 << 1) | U_XFUNCS_;
const int     ParametricBoundary::U_SFUNCS_    = (1 << 2);
const int     ParametricBoundary::U_SGRADS_    = (1 << 3) | U_SFUNCS_;

const int     ParametricBoundary::U_FUNCS_     = U_XFUNCS_  |
                                                 U_SFUNCS_;
const int     ParametricBoundary::U_GRADS_     = U_XGRADS_  |
                                                 U_SGRADS_;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


ParametricBoundary::ParametricBoundary ()
{
  rank_        = -1_idx;
  ipCount_     = -1_idx;
  nodeCount_   = -1_idx;
  funcCount_   = -1_idx;
  vertexCount_ = -1_idx;
}


ParametricBoundary::ParametricBoundary

  ( const String&         name,
    const Matrix&         ischeme,
    const Ref<StdShape>&  xshape,
    const Ref<StdShape>&  sshape ) :

    Super    (    name ),
    xshape_  (  xshape ),
    sshape_  (  sshape ),
    ischeme_ ( ischeme )

{
  JEM_PRECHECK2 ( xshape, "NULL geometric shape functions" );

  if ( ! sshape_ )
  {
    sshape_ = xshape_;
  }

  init_ ();
}


ParametricBoundary::~ParametricBoundary ()
{}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void ParametricBoundary::readFrom ( ObjectInput& in )
{
  decode ( in, myName_, ischeme_, xshape_, sshape_ );
  init_  ();
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void ParametricBoundary::writeTo ( ObjectOutput& out ) const
{
  encode ( out, myName_, ischeme_, xshape_, sshape_ );
}


//-----------------------------------------------------------------------
//   localRank
//-----------------------------------------------------------------------


idx_t ParametricBoundary::localRank () const
{
  return rank_;
}


//-----------------------------------------------------------------------
//   nodeCount
//-----------------------------------------------------------------------


idx_t ParametricBoundary::nodeCount () const
{
  return nodeCount_;
}


//-----------------------------------------------------------------------
//   vertexCount
//-----------------------------------------------------------------------


idx_t ParametricBoundary::vertexCount () const
{
  return vertexCount_;
}


//-----------------------------------------------------------------------
//   ipointCount
//-----------------------------------------------------------------------


idx_t ParametricBoundary::ipointCount () const
{
  return ipCount_;
}


//-----------------------------------------------------------------------
//   shapeFuncCount
//-----------------------------------------------------------------------


idx_t ParametricBoundary::shapeFuncCount () const
{
  return funcCount_;
}


//-----------------------------------------------------------------------
//   getGeometry
//-----------------------------------------------------------------------


String ParametricBoundary::getGeometry () const
{
  return xshape_->getGeometry ();
}


//-----------------------------------------------------------------------
//   getLocalWeights
//-----------------------------------------------------------------------


Vector ParametricBoundary::getLocalWeights () const
{
  return iweights_;
}


//-----------------------------------------------------------------------
//   getVertexCoords
//-----------------------------------------------------------------------


Matrix ParametricBoundary::getVertexCoords () const
{
  return vertices_.coords;
}


//-----------------------------------------------------------------------
//   getIntegrationScheme
//-----------------------------------------------------------------------


Matrix ParametricBoundary::getIntegrationScheme () const
{
  return ischeme_;
}


//-----------------------------------------------------------------------
//   getGlobalIntegrationPoints
//-----------------------------------------------------------------------


void ParametricBoundary::getGlobalIntegrationPoints

  ( const Matrix& x,
    const Matrix& c ) const

{
  updateFuncs_ ( ipoints_ );

  matmul ( x, c, ipoints_.xfuncs );
}


//-----------------------------------------------------------------------
//   getGlobalPoint
//-----------------------------------------------------------------------


void ParametricBoundary::getGlobalPoint

  ( const Vector& x,
    const Vector& u,
    const Matrix& c ) const

{
  const Vector&  xsf = scratch_.xfuncs;

  xshape_->evalShapeFunctions ( xsf, u );

  matmul ( x, c, xsf );
}


//-----------------------------------------------------------------------
//   getLocalPoint
//-----------------------------------------------------------------------


bool ParametricBoundary::getLocalPoint

  ( const Vector&  u,
    const Vector&  x,
    double         eps,
    const Matrix&  c ) const

{
  using jem::maxOf;
  using jem::numeric::LUSolver;

  const StdShape&  xsh    = * xshape_;
  const double     eps2   = eps * eps;
  const idx_t      rank   = rank_;

  const Vector&    xsf    = scratch_.xfuncs;
  const Matrix&    xgrads = scratch_.xgrads;

  Matrix           gt     = xgrads.transpose ();

  IdxVector        iperm  ( rank );

  Matrix           buf1   ( rank + 1, rank + 1 );
  Matrix           buf2   ( rank,     rank + 4 );

  Matrix           gx     ( buf1[slice(BEGIN,rank)] );
  Matrix           px     ( buf2[slice(BEGIN,rank)] );
  Vector           dx     ( buf1[rank + 0] );
  Vector           du     ( buf2[rank + 0] );
  Vector           dv     ( buf2[rank + 1] );
  Vector           pxii   ( buf2[rank + 2] );
  Vector           scale  ( buf2[rank + 3] );

  double           ddx, ddx0;
  double           delta;
  double           tmp;
  int              sign;
  int              iiter;


  // Execute a Newton-Raphson procedure to find the local
  // point. Start in the geometric center of the shape.

  iiter = 0;
  delta = maxOf ( delta );
  ddx0  = 0.0;
  u     = center_;

  while ( true )
  {
    xsh.evalShapeGradients ( xsf, xgrads, u );
    matmul                 ( dx,  c,    xsf );

    ddx = 0.0;

    for ( idx_t k = 0; k <= rank; k++ )
    {
      tmp   = x[k] - dx[k];
      ddx  += tmp * tmp;
      dx[k] = tmp;
    }

    if ( iiter == 0 )
    {
      ddx0 = ddx;
    }

    if ( delta < NR_DELTA_ * eps2 ||
         ddx   < NR_ALPHA_ * eps2 ||
         ddx   > NR_BETA_  * ddx0 ||
         iiter > MAX_NR_ITER_ )
    {
      break;
    }

    matmul ( gx, c, gt );

    // Compute the following projections:
    //   px = gx^t * gx
    //   dv = gx^t * dx

    for ( idx_t i = 0; i < rank; i++ )
    {
      tmp = 0.0;

      for ( idx_t j = 0; j <= rank; j++ )
      {
        tmp += gx(j,i) * dx[j];
      }

      dv[i] = tmp;

      for ( idx_t j = 0; j < rank; j++ )
      {
        tmp = 0.0;

        for ( idx_t k = 0; k <= rank; k++ )
        {
          tmp += gx(k,i) * gx(k,j);
        }

        px(i,j) = tmp;
      }

      pxii[i] = px(i,i);
    }

    // Solve the projected system: gx^t * gx * du = dv.

    if ( ! LUSolver::factor( px, scale, iperm, sign ) )
    {
      break;
    }

    LUSolver::solve ( du, px, dv, iperm );

    // Compute the next local and global points.

    delta = 0.0;

    for ( idx_t k = 0; k < rank; k++ )
    {
      tmp    = du[k];
      delta += pxii[k] * tmp * tmp;
      u[k]  += tmp;
    }

    iiter++;
  }

  if ( xsh.clipPoint( u ) )
  {
    xsh.evalShapeFunctions ( xsf, u );

    matmul ( dx, c, xsf );

    ddx = 0.0;

    for ( idx_t k = 0; k <= rank; k++ )
    {
      tmp  = x[k] - dx[k];
      ddx += tmp * tmp;
    }
  }

  return (ddx < eps2);
}


//-----------------------------------------------------------------------
//   containsLocalPoint
//-----------------------------------------------------------------------


bool ParametricBoundary::containsLocalPoint

  ( const Vector& u ) const

{
  return xshape_->containsPoint ( u );
}


//-----------------------------------------------------------------------
//   getShapeFunctions
//-----------------------------------------------------------------------


Matrix ParametricBoundary::getShapeFunctions () const
{
  updateFuncs_ ( ipoints_ );

  return ipoints_.sfuncs;
}


//-----------------------------------------------------------------------
//   getVertexFunctions
//-----------------------------------------------------------------------


Matrix ParametricBoundary::getVertexFunctions () const
{
  updateFuncs_ ( vertices_ );

  return vertices_.sfuncs;
}


//-----------------------------------------------------------------------
//   evalShapeFunctions
//-----------------------------------------------------------------------


void ParametricBoundary::evalShapeFunctions

  ( const Vector&  h,
    const Vector&  u ) const

{
  sshape_->evalShapeFunctions ( h, u );
}


//-----------------------------------------------------------------------
//   getNormals
//-----------------------------------------------------------------------


void ParametricBoundary::getNormals

  ( const Matrix&  q,
    const Vector&  w,
    const Matrix&  c ) const

{
  updateGrads_ (           ipoints_ );
  calcNormals  ( q, &w, c, ipoints_ );

JEM_IVDEP

  for ( idx_t ip = 0; ip < ipCount_; ip++ )
  {
    w[ip] = iweights_[ip] * std::fabs ( w[ip] );
  }
}


//-----------------------------------------------------------------------
//   getVertexNormals
//-----------------------------------------------------------------------


void ParametricBoundary::getVertexNormals

  ( const Matrix&  q,
    const Matrix&  c ) const

{
  updateGrads_ (          vertices_ );
  calcNormals  ( q, 0, c, vertices_ );
}


//-----------------------------------------------------------------------
//   getXShape & getSShape
//-----------------------------------------------------------------------


StdShape* ParametricBoundary::getXShape () const
{
  return xshape_.get ();
}


StdShape* ParametricBoundary::getSShape () const
{
  return sshape_.get ();
}


//-----------------------------------------------------------------------
//   getExtByID
//-----------------------------------------------------------------------


void* ParametricBoundary::getExtByID ( ExtensionID id ) const
{
  void*  ext = xshape_->getExtByID ( id );

  if ( ! ext && sshape_ != xshape_ )
  {
    ext = sshape_->getExtByID ( id );
  }

  return ext;
}

//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void ParametricBoundary::init_ ()
{
  using jem::util::connect;

  const StdShape&  xsh = * xshape_;
  const StdShape&  ssh = * sshape_;

  String  context = getContext ();


  ParametricUtils::checkShapes  ( context, xsh, ssh );
  ParametricUtils::checkIScheme ( context, xsh, ischeme_ );

  iweights_      .ref ( ischeme_(0,ALL) );
  ipoints_.coords.ref ( ischeme_(slice(1,END),ALL) );

  rank_        = xsh.rank           ();
  ipCount_     = iweights_.size     ();
  nodeCount_   = xsh.shapeFuncCount ();
  funcCount_   = ssh.shapeFuncCount ();
  vertexCount_ = xsh.vertexCount    ();

  vertices_.coords.ref ( xsh.getVertexCoords() );

  center_.resize ( rank_ );

  center_ = 0.0;

  for ( idx_t j = 0; j < vertexCount_; j++ )
  {
    for ( idx_t i = 0; i < rank_; i++ )
    {
      center_[i] += vertices_.coords(i,j);
    }
  }

  if ( vertexCount_ > 0 )
  {
    center_ *= 1.0 / (double) vertexCount_;
  }

  ipoints_ .updated = U_NONE_;
  vertices_.updated = U_NONE_;

  scratch_.xfuncs.resize (        nodeCount_ );
  scratch_.xgrads.resize ( rank_, nodeCount_ );

  if ( sshape_ == xshape_ )
  {
    scratch_.sfuncs.ref ( scratch_.xfuncs );

    connect ( xshape_->changeEvent, this, &Self::shapeChanged_ );
  }
  else
  {
    scratch_.sfuncs.resize ( funcCount_ );

    connect ( xshape_->changeEvent, this, &Self::xshapeChanged_ );
    connect ( sshape_->changeEvent, this, &Self::sshapeChanged_ );
  }
}


//-----------------------------------------------------------------------
//   shapeChanged_
//-----------------------------------------------------------------------


void ParametricBoundary::shapeChanged_ ()
{
  ipoints_ .updated = U_NONE_;
  vertices_.updated = U_NONE_;

  changeEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   xshapeChanged_
//-----------------------------------------------------------------------


void ParametricBoundary::xshapeChanged_ ()
{
  ipoints_ .updated &= ~U_XGRADS_;
  vertices_.updated &= ~U_XGRADS_;

  changeEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   sshapeChanged_
//-----------------------------------------------------------------------


void ParametricBoundary::sshapeChanged_ ()
{
  ipoints_ .updated &= ~U_SGRADS_;
  vertices_.updated &= ~U_SGRADS_;

  changeEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   doUpdateFuncs_
//-----------------------------------------------------------------------


void ParametricBoundary::doUpdateFuncs_ ( PointSet_& points ) const
{
  const idx_t  pcount = points.coords.size(1);

  // Allocate memory if needed.

  if ( points.xfuncs.size(0) == 0 )
  {
    points.xfuncs.resize ( nodeCount_, pcount );

    if ( sshape_ == xshape_ )
    {
      points.sfuncs.ref ( points.xfuncs );
    }
    else
    {
      points.sfuncs.resize ( funcCount_, pcount );
    }
  }

  // Update the shape functions.

  if ( (points.updated & U_XFUNCS_) != U_XFUNCS_ )
  {
    for ( idx_t ip = 0; ip < pcount; ip++ )
    {
      xshape_->evalShapeFunctions ( points.xfuncs[ip],
                                    points.coords[ip] );
    }

    points.updated |= U_XFUNCS_;

    if ( sshape_ == xshape_ )
    {
      points.updated |= U_SFUNCS_;
    }
  }

  if ( (points.updated & U_SFUNCS_) != U_SFUNCS_ )
  {
    for ( idx_t ip = 0; ip < pcount; ip++ )
    {
      sshape_->evalShapeFunctions ( points.sfuncs[ip],
                                    points.coords[ip] );
    }

    points.updated |= U_SFUNCS_;
  }
}


//-----------------------------------------------------------------------
//   doUpdateGrads_
//-----------------------------------------------------------------------


void ParametricBoundary::doUpdateGrads_ ( PointSet_& points ) const
{
  const idx_t  pcount = points.coords.size(1);

  // Allocate memory if needed.

  if ( points.xgrads.size(0) == 0 )
  {
    points.xfuncs.resize (        nodeCount_, pcount );
    points.xgrads.resize ( rank_, nodeCount_, pcount );

    if ( sshape_ == xshape_ )
    {
      points.sfuncs.ref ( points.xfuncs );
      points.sgrads.ref ( points.xgrads );
    }
    else
    {
      points.sfuncs.resize (        funcCount_, pcount );
      points.sgrads.resize ( rank_, funcCount_, pcount );
    }
  }

  // Update the shape functions and the gradients.

  if ( (points.updated & U_XGRADS_) != U_XGRADS_ )
  {
    for ( idx_t ip = 0; ip < pcount; ip++ )
    {
      xshape_->evalShapeGradients ( points.xfuncs[ip],
                                    points.xgrads[ip],
                                    points.coords[ip] );
    }

    points.updated |= U_XGRADS_;

    if ( sshape_ == xshape_ )
    {
      points.updated |= U_SGRADS_;
    }
  }

  if ( (points.updated & U_SGRADS_) != U_SGRADS_ )
  {
    for ( idx_t ip = 0; ip < pcount; ip++ )
    {
      sshape_->evalShapeGradients ( points.sfuncs[ip],
                                    points.sgrads[ip],
                                    points.coords[ip] );
    }

    points.updated |= U_SGRADS_;
  }
}


JIVE_END_PACKAGE( geom )
