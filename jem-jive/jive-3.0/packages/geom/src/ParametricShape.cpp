
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
#include <jem/base/assert.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/array/operators.h>
#include <jem/base/array/utilities.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/Event.h>
#include <jem/numeric/algebra/matmul.h>
#include <jem/numeric/algebra/LUSolver.h>
#include <jive/util/error.h>
#include <jive/geom/error.h>
#include <jive/geom/utilities.h>
#include <jive/geom/StdBoundary.h>
#include <jive/geom/ShapeBoundary.h>
#include <jive/geom/BoundaryShape.h>
#include <jive/geom/ParametricUtils.h>
#include <jive/geom/ParametricShape.h>


JEM_DEFINE_SERIABLE_CLASS ( jive::geom::ParametricShape );


JIVE_BEGIN_PACKAGE( geom )


using jem::checkedCast;
using jem::numeric::matmul;
using jem::numeric::LUSolver;


//=======================================================================
//   class ParametricShape
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int     ParametricShape::MAX_NR_ITER_ = 20;
const double  ParametricShape::NR_ALPHA_    = 1.0e-2;
const double  ParametricShape::NR_BETA_     = 1.0e20;

const int     ParametricShape::U_NONE_      = 0;

const int     ParametricShape::U_XFUNCS_    = (1 << 0);
const int     ParametricShape::U_XGRADS_    = (1 << 1) | U_XFUNCS_;
const int     ParametricShape::U_XGRADS2_   = (1 << 2) | U_XGRADS_;
const int     ParametricShape::U_SFUNCS_    = (1 << 3);
const int     ParametricShape::U_SGRADS_    = (1 << 4) | U_SFUNCS_;
const int     ParametricShape::U_SGRADS2_   = (1 << 5) | U_SGRADS_;

const int     ParametricShape::U_FUNCS_     = U_XFUNCS_  |
                                              U_SFUNCS_;
const int     ParametricShape::U_GRADS_     = U_XGRADS_  |
                                              U_SGRADS_;
const int     ParametricShape::U_GRADS2_    = U_XGRADS2_ |
                                              U_SGRADS2_;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


ParametricShape::ParametricShape ()
{
  rank_        = -1_idx;
  rank2_       = -1_idx;
  ipCount_     = -1_idx;
  nodeCount_   = -1_idx;
  funcCount_   = -1_idx;
  vertexCount_ = -1_idx;
}


ParametricShape::ParametricShape

  ( const String&         name,
    const Matrix&         ischeme,
    const Ref<StdShape>&  xshape,
    const Ref<StdShape>&  sshape ) :

    Super    (    name ),
    xshape_  (  xshape ),
    sshape_  (  sshape ),
    ischeme_ ( ischeme )

{
  JEM_PRECHECK2 ( xshape, "NULL geometric Shape" );

  if ( ! sshape_ )
  {
    sshape_ = xshape_;
  }

  init_ ();
}


ParametricShape::ParametricShape

  ( const String&         name,
    const Matrix&         ischeme,
    const Boundary&       boundary,
    const Ref<StdShape>&  xshape,
    const Ref<StdShape>&  sshape ) :

    Super     (     name ),
    xshape_   (   xshape ),
    sshape_   (   sshape ),
    ischeme_  (  ischeme ),
    boundary_ ( boundary )

{
  JEM_PRECHECK2 ( xshape, "NULL geometric Shape" );

  if ( ! sshape_ )
  {
    sshape_ = xshape_;
  }

  init_ ();
}


ParametricShape::~ParametricShape ()
{}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void ParametricShape::readFrom ( ObjectInput& in )
{
  decode ( in, myName_, ischeme_, boundary_, xshape_, sshape_ );
  init_  ();
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void ParametricShape::writeTo ( ObjectOutput& out ) const
{
  encode ( out, myName_, ischeme_, boundary_, xshape_, sshape_ );
}


//-----------------------------------------------------------------------
//   localRank
//-----------------------------------------------------------------------


idx_t ParametricShape::localRank () const
{
  return rank_;
}


//-----------------------------------------------------------------------
//   nodeCount
//-----------------------------------------------------------------------


idx_t ParametricShape::nodeCount () const
{
  return nodeCount_;
}


//-----------------------------------------------------------------------
//   vertexCount
//-----------------------------------------------------------------------


idx_t ParametricShape::vertexCount () const
{
  return vertexCount_;
}


//-----------------------------------------------------------------------
//   ipointCount
//-----------------------------------------------------------------------


idx_t ParametricShape::ipointCount () const
{
  return ipCount_;
}


//-----------------------------------------------------------------------
//   shapeFuncCount
//-----------------------------------------------------------------------


idx_t ParametricShape::shapeFuncCount () const
{
  return funcCount_;
}


//-----------------------------------------------------------------------
//   boundaryCount
//-----------------------------------------------------------------------


idx_t ParametricShape::boundaryCount () const
{
  return bpoints_.size ();
}


//-----------------------------------------------------------------------
//   getGeometry
//-----------------------------------------------------------------------


String ParametricShape::getGeometry () const
{
  return xshape_->getGeometry ();
}


//-----------------------------------------------------------------------
//   getBoundaryShape
//-----------------------------------------------------------------------


BoundaryShape* ParametricShape::getBoundaryShape ( idx_t ibnd ) const
{
  return boundary_.shapes[ibnd].get ();
}


//-----------------------------------------------------------------------
//   getBoundaryNodes
//-----------------------------------------------------------------------


IdxVector ParametricShape::getBoundaryNodes ( idx_t ibnd ) const
{
  return boundary_.topology.getColumnIndices ( ibnd );
}


//-----------------------------------------------------------------------
//   getBoundaryTopology
//-----------------------------------------------------------------------


InternalShape::Topology ParametricShape::getBoundaryTopology () const
{
  return boundary_.topology;
}


//-----------------------------------------------------------------------
//   getBoundaryCoords
//-----------------------------------------------------------------------


void ParametricShape::getBoundaryCoords

  ( const Matrix&  x,
    idx_t          ibnd,
    const Matrix&  c ) const

{
  using jive::util::indexError;

  const Topology&  topo = boundary_.topology;

  JEM_ASSERT2 ( topo.isContiguous(),
                "non-contiguous boundary Topology" );

  if ( ibnd < 0 || ibnd >= topo.size(0) )
  {
    indexError ( getContext(), "boundary", ibnd, topo.size(0) );
  }

  const idx_t* JEM_RESTRICT  offsets = topo.getOffsetPtr ();
  const idx_t* JEM_RESTRICT  indices = topo.getIndexPtr  ();

  const idx_t  first = offsets[ibnd];
  const idx_t  last  = offsets[ibnd + 1];
  const idx_t  rank  = rank_;
  const idx_t  count = last - first;


  for ( idx_t j = 0; j < count; j++ )
  {
    idx_t  k = indices[j + first];

JEM_IVDEP

    for ( idx_t i = 0; i < rank; i++ )
    {
      x(i,j) = c(i,k);
    }
  }
}


//-----------------------------------------------------------------------
//   mapBoundaryPoint
//-----------------------------------------------------------------------


void ParametricShape::mapBoundaryPoint

  ( const Vector&  v,
    idx_t          ibnd,
    const Vector&  u ) const

{
  using jive::util::indexError;

  if ( ibnd < 0 || ibnd >= boundary_.bounds.size() )
  {
    indexError ( getContext(), "boundary",
                 ibnd, boundary_.bounds.size() );
  }

  boundary_.bounds[ibnd]->mapBoundaryPoint ( v, u );
}


//-----------------------------------------------------------------------
//   getLocalWeights
//-----------------------------------------------------------------------


Vector ParametricShape::getLocalWeights () const
{
  return iweights_;
}


//-----------------------------------------------------------------------
//   getVertexCoords
//-----------------------------------------------------------------------


Matrix ParametricShape::getVertexCoords () const
{
  return vertices_.coords;
}


//-----------------------------------------------------------------------
//   getIntegrationScheme
//-----------------------------------------------------------------------


Matrix ParametricShape::getIntegrationScheme () const
{
  return ischeme_;
}


//-----------------------------------------------------------------------
//   getGlobalIntegrationPoints
//-----------------------------------------------------------------------


void ParametricShape::getGlobalIntegrationPoints

  ( const Matrix& x,
    const Matrix& c ) const

{
  updateFuncs_ ( ipoints_ );

  matmul ( x, c, ipoints_.xfuncs );
}


//-----------------------------------------------------------------------
//   getIntegrationWeights
//-----------------------------------------------------------------------


void ParametricShape::getIntegrationWeights

  ( const Vector&  w,
    const Matrix&  c ) const

{
  const Cubix&  xgrads = ipoints_.xgrads;

  Matrix        ct     = c.transpose ();

  Matrix        buf    ( rank_, rank_ + 1 );

  Matrix        ja     ( buf[slice(BEGIN,rank_)] );
  Vector        scale  ( buf[rank_] );

  IdxVector     iperm  ( rank_ );

  int           sign;


  updateGrads_ ( ipoints_ );

  for ( idx_t ip = 0; ip < ipCount_; ip++ )
  {
    matmul ( ja, xgrads[ip], ct );

    if ( LUSolver::factor( ja, scale, iperm, sign ) )
    {
      w[ip] = iweights_[ip] * std::fabs ( LUSolver::det( ja ) );
    }
    else
    {
      w[ip] = 0.0;
    }
  }
}


//-----------------------------------------------------------------------
//   getGlobalPoint
//-----------------------------------------------------------------------


void ParametricShape::getGlobalPoint

  ( const Vector& x,
    const Vector& u,
    const Matrix& c ) const

{
  xshape_->evalShapeFunctions ( scratch_.xfuncs, u );

  matmul ( x, c, scratch_.xfuncs );
}


//-----------------------------------------------------------------------
//   getLocalPoint
//-----------------------------------------------------------------------


bool ParametricShape::getLocalPoint

  ( const Vector&  u,
    const Vector&  x,
    double         eps,
    const Matrix&  c ) const

{
  const StdShape&  xsh    = * xshape_;
  const double     eps2   = eps * eps;
  const idx_t      rank   = rank_;

  const Vector&    xsf    = scratch_.xfuncs;
  const Matrix&    xgrads = scratch_.xgrads;

  Matrix           gt     = xgrads.transpose ();

  IdxVector        iperm  ( rank );

  Matrix           buf    ( rank, rank + 3 );

  Matrix           gx     ( buf[slice(BEGIN,rank)] );
  Vector           dx     ( buf[rank + 0] );
  Vector           du     ( buf[rank + 1] );
  Vector           scale  ( buf[rank + 2] );

  double           ddx, ddx0;
  double           tmp;
  int              sign;
  int              iiter;


  // Execute a Newton-Raphson procedure to find the local
  // point. Start in the geometric center of the shape.

  iiter = 0;
  ddx0  = 0.0;
  u     = center_;

  while ( true )
  {
    xsh.evalShapeGradients ( xsf, xgrads, u );
    matmul                 ( dx,  c,    xsf );

    ddx = 0.0;

    for ( idx_t i = 0; i < rank; i++ )
    {
      tmp   = x[i] - dx[i];
      ddx  += tmp * tmp;
      dx[i] = tmp;
    }

    if ( iiter == 0 )
    {
      ddx0 = ddx;
    }

    if ( ddx   < NR_ALPHA_ * eps2 ||
         ddx   > NR_BETA_  * ddx0 ||
         iiter > MAX_NR_ITER_ )
    {
      break;
    }

    matmul ( gx, c, gt );

    if ( ! LUSolver::factor( gx, scale, iperm, sign ) )
    {
      break;
    }

    LUSolver::solve ( du, gx, dx, iperm );

    for ( idx_t i = 0; i < rank; i++ )
    {
      u[i] += du[i];
    }

    iiter++;
  }

  if ( xsh.clipPoint( u ) )
  {
    xsh.evalShapeFunctions ( xsf, u );
    matmul                 ( dx,  c, xsf );

    ddx = 0.0;

    for ( idx_t i = 0; i < rank; i++ )
    {
      tmp  = x[i] - dx[i];
      ddx += tmp * tmp;
    }
  }

  return (ddx < eps2);
}


//-----------------------------------------------------------------------
//   containsLocalPoint
//-----------------------------------------------------------------------


bool ParametricShape::containsLocalPoint

  ( const Vector& u ) const

{
  return xshape_->containsPoint ( u );
}


//-----------------------------------------------------------------------
//   getShapeFunctions
//-----------------------------------------------------------------------


Matrix ParametricShape::getShapeFunctions () const
{
  updateFuncs_ ( ipoints_ );

  return ipoints_.sfuncs;
}


//-----------------------------------------------------------------------
//   getVertexFunctions
//-----------------------------------------------------------------------


Matrix ParametricShape::getVertexFunctions () const
{
  updateFuncs_ ( vertices_ );

  return vertices_.sfuncs;
}


//-----------------------------------------------------------------------
//   evalShapeFunctions
//-----------------------------------------------------------------------


void ParametricShape::evalShapeFunctions

  ( const Vector&  h,
    const Vector&  u ) const

{
  sshape_->evalShapeFunctions ( h, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradients
//-----------------------------------------------------------------------


void ParametricShape::evalShapeGradients

  ( const Matrix&  g,
    const Vector&  u,
    const Matrix&  c ) const

{
  const Matrix&  xgrads = scratch_.xgrads;
  const Matrix&  sgrads = scratch_.sgrads;

  Matrix         buf    ( rank_, rank_ + 1 );

  Matrix         ja     ( buf[slice(BEGIN,rank_)] );
  Vector         scale  ( buf[rank_] );

  IdxVector      iperm  ( rank_ );

  int            sign;


  xshape_->evalShapeGradients ( scratch_.xfuncs, xgrads, u );

  if ( sshape_ != xshape_ )
  {
    sshape_->evalShapeGradients ( scratch_.sfuncs, sgrads, u );
  }

  matmul ( ja, xgrads, c.transpose() );

  if ( ! LUSolver::factor( ja, scale, iperm, sign ) )
  {
    singularMatrixError ( getContext(), "Jacobi" );
  }

  LUSolver::solve ( g, ja, sgrads, iperm );
}


//-----------------------------------------------------------------------
//   getShapeGradients
//-----------------------------------------------------------------------


void ParametricShape::getShapeGradients

  ( const Cubix&   g,
    const Vector&  w,
    const Matrix&  c ) const

{
  updateGrads_  (           ipoints_ );
  calcGradients ( g, &w, c, ipoints_ );

JEM_IVDEP

  for ( idx_t ip = 0; ip < ipCount_; ip++ )
  {
    w[ip] = iweights_[ip] * std::fabs( w[ip] );
  }
}


//-----------------------------------------------------------------------
//   getVertexGradients
//-----------------------------------------------------------------------


void ParametricShape::getVertexGradients

  ( const Cubix&   g,
    const Matrix&  c ) const

{
  updateGrads_  (          vertices_ );
  calcGradients ( g, 0, c, vertices_ );
}


//-----------------------------------------------------------------------
//   getBoundaryGradients
//-----------------------------------------------------------------------


void ParametricShape::getBoundaryGradients

  ( const Cubix&   g,
    idx_t          ibnd,
    const Matrix&  c ) const

{
  const PointSet_&  bpts = bpoints_[ibnd];

  updateGrads_  (          bpts );
  calcGradients ( g, 0, c, bpts );
}


//-----------------------------------------------------------------------
//   evalShapeGradGrads
//-----------------------------------------------------------------------


void ParametricShape::evalShapeGradGrads

  ( const Matrix&  g2,
    const Matrix&  g,
    const Vector&  u,
    const Matrix&  c ) const

{
  const Vector&  xfuncs  = scratch_.xfuncs;
  const Matrix&  xgrads  = scratch_.xgrads;
  const Matrix&  xgrads2 = scratch_.xgrads2;
  const Vector&  sfuncs  = scratch_.sfuncs;
  const Matrix&  sgrads  = scratch_.sgrads;
  const Matrix&  sgrads2 = scratch_.sgrads2;

  IdxVector      ibuf  ( rank_ + rank2_ );

  IdxVector      iperm ( ibuf[slice(BEGIN,rank_)] );
  IdxVector      jperm ( ibuf[slice(rank_,END)]   );

  Matrix         ct    ( c.transpose()  );

  Matrix         ja    ( rank_,  rank_  );
  Matrix         ja2   ( rank2_, rank_  );
  Matrix         qu    ( rank2_, rank2_ );

  Vector         scale ( rank2_ );

  int            sign;


  xshape_->evalShapeGradGrads ( xfuncs, xgrads, xgrads2, u );

  if ( sshape_ != xshape_ )
  {
    sshape_->evalShapeGradGrads ( sfuncs, sgrads, sgrads2, u );
  }

  matmul ( ja,  xgrads,  ct );
  matmul ( ja2, xgrads2, ct );

  getQujobi ( qu, ja );

  if ( ! LUSolver::factor( ja, scale, iperm, sign ) )
  {
    singularMatrixError ( getContext(), "Jacobi" );
  }

  if ( ! LUSolver::factor( qu, scale, jperm, sign ) )
  {
    singularMatrixError ( getContext(), "Qujobi" );
  }

  LUSolver::solve ( g, ja, sgrads, iperm );

  matmul ( g2, ja2, g );

  sgrads2 -= g2;

  LUSolver::solve ( g2, qu, sgrads2, jperm );
}


//-----------------------------------------------------------------------
//   getShapeGradGrads
//-----------------------------------------------------------------------


void ParametricShape::getShapeGradGrads

  ( const Cubix&   g2,
    const Cubix&   g,
    const Vector&  w,
    const Matrix&  c ) const

{
  updateGrads2_ (               ipoints_ );
  calcGradGrads ( g2, g, &w, c, ipoints_ );

JEM_IVDEP

  for ( idx_t ip = 0; ip < ipCount_; ip++ )
  {
    w[ip] = iweights_[ip] * std::fabs ( w[ip] );
  }
}


//-----------------------------------------------------------------------
//   getVertexGradGrads
//-----------------------------------------------------------------------


void ParametricShape::getVertexGradGrads

  ( const Cubix&   g2,
    const Cubix&   g,
    const Matrix&  c ) const

{
  updateGrads2_ (              vertices_ );
  calcGradGrads ( g2, g, 0, c, vertices_ );
}


//-----------------------------------------------------------------------
//   getBoundaryGradGrads
//-----------------------------------------------------------------------


void ParametricShape::getBoundaryGradGrads

  ( const Cubix&   g2,
    const Cubix&   g,
    idx_t          ibnd,
    const Matrix&  c )  const

{
  const PointSet_&  bpts = bpoints_[ibnd];

  updateGrads2_ (              bpts );
  calcGradGrads ( g2, g, 0, c, bpts );
}


//-----------------------------------------------------------------------
//   getXShape & getSShape
//-----------------------------------------------------------------------


StdShape* ParametricShape::getXShape () const
{
  return xshape_.get ();
}


StdShape* ParametricShape::getSShape () const
{
  return sshape_.get ();
}


//-----------------------------------------------------------------------
//   calcGradients
//-----------------------------------------------------------------------


void ParametricShape::calcGradients

  ( const Cubix&     g,
    const Vector*    w,
    const Matrix&    c,
    const PointSet&  points ) const

{
  const Cubix&  xgrads = points.xgrads;
  const Cubix&  sgrads = points.sgrads;

  const idx_t   pcount = xgrads.size(2);

  Matrix        ct     = c.transpose ();

  Matrix        buf    ( rank_, rank_ + 1 );

  Matrix        ja     ( buf[slice(BEGIN,rank_)] );
  Vector        scale  ( buf[rank_] );

  IdxVector     iperm  ( rank_ );

  int           sign;


  for ( idx_t ip = 0; ip < pcount; ip++ )
  {
    matmul ( ja, xgrads[ip], ct );

    if ( ! LUSolver::factor( ja, scale, iperm, sign ) )
    {
      singularMatrixError ( getContext(), "Jacobi" );
    }

    LUSolver::solve ( g[ip], ja, sgrads[ip], iperm );

    if ( w )
    {
      (*w)[ip] = std::fabs ( LUSolver::det( ja ) );
    }
  }
}


//-----------------------------------------------------------------------
//   calcGradGrads
//-----------------------------------------------------------------------


void ParametricShape::calcGradGrads

  ( const Cubix&     g2,
    const Cubix&     g,
    const Vector*    w,
    const Matrix&    c,
    const PointSet&  points ) const

{
  const Cubix&   xgrads  = points.xgrads;
  const Cubix&   xgrads2 = points.xgrads2;
  const Cubix&   sgrads  = points.sgrads;
  const Cubix&   sgrads2 = points.sgrads2;
  const Matrix&  tgrads2 = scratch_.sgrads2;

  const idx_t    pcount  = xgrads.size(2);

  IdxVector      ibuf   ( rank_ + rank2_ );

  IdxVector      iperm  ( ibuf[slice(BEGIN,rank_)] );
  IdxVector      jperm  ( ibuf[slice(rank_,END)]   );

  Matrix         ct     ( c.transpose()  );

  Matrix         ja     ( rank_,  rank_  );
  Matrix         ja2    ( rank2_, rank_  );
  Matrix         qu     ( rank2_, rank2_ );

  Vector         scale  ( rank2_ );

  int            sign;


  for ( idx_t ip = 0; ip < pcount; ip++ )
  {
    matmul ( ja,  xgrads [ip], ct );
    matmul ( ja2, xgrads2[ip], ct );

    getQujobi ( qu, ja );

    if ( ! LUSolver::factor( ja, scale, iperm, sign ) )
    {
      singularMatrixError ( getContext(), "Jacobi" );
    }

    if ( ! LUSolver::factor( qu, scale, jperm, sign ) )
    {
      singularMatrixError ( getContext(), "Qujobi" );
    }

    LUSolver::solve ( g[ip],   ja,  sgrads[ip], iperm );
    matmul          ( tgrads2, ja2, g[ip] );

    tgrads2 = sgrads2[ip] - tgrads2;

    LUSolver::solve ( g2[ip], qu, tgrads2, jperm );

    if ( w )
    {
      (*w)[ip] = std::fabs ( LUSolver::det( ja ) );
    }
  }
}


//-----------------------------------------------------------------------
//   getExtByID
//-----------------------------------------------------------------------


void* ParametricShape::getExtByID ( ExtensionID id ) const
{
  if ( id == Grad2Extension::ID )
  {
    Grad2Extension*  ext = const_cast<Self*> ( this );

    return ext;
  }
  else
  {
    void*  ext = xshape_->getExtByID ( id );

    if ( ! ext && sshape_ != xshape_ )
    {
      ext = sshape_->getExtByID ( id );
    }

    return ext;
  }
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void ParametricShape::init_ ()
{
  using jem::util::connect;

  Boundary&    bnd      = boundary_;
  StdShape&    xsh      = * xshape_;
  StdShape&    ssh      = * sshape_;

  const idx_t  bndCount = bnd.shapes.size ();

  String       context  = getContext ();


  if ( ! bnd.topology.isContiguous() )
  {
    bnd.topology = bnd.topology.clone ();
  }

  ParametricUtils::checkShapes  ( context, xsh, ssh );
  ParametricUtils::checkIScheme ( context, xsh, ischeme_ );

  iweights_      .ref ( ischeme_(0,ALL) );
  ipoints_.coords.ref ( ischeme_(slice(1,END),ALL) );

  if ( bndCount > 0 )
  {
    ParametricUtils::checkBounds ( context, xsh, ssh, bnd );
  }

  rank_        = xsh.rank ();
  rank2_       = (rank_ * rank_ + rank_) / 2;
  ipCount_     = iweights_ .size    ();
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

  bpoints_.resize ( bndCount );

  for ( idx_t ibnd = 0; ibnd < bndCount; ibnd++ )
  {
    const BShape&       bsh     = * bnd.shapes[ibnd];
    const StdBoundary&  sbsh    = * bnd.bounds[ibnd];
    PointSet_&          bpts    = bpoints_[ibnd];
    Matrix              ischeme = bsh.getIntegrationScheme ();
    const idx_t         ipCount = ischeme.size (1);

    bpts.coords.resize ( rank_, ipCount );

    for ( idx_t ip = 0; ip < ipCount; ip++ )
    {
      sbsh.mapBoundaryPoint ( bpts.coords[ip],
                              ischeme(slice(1,END),ip) );
    }

    bpts.updated = U_NONE_;
  }

  scratch_.xfuncs .resize   (         nodeCount_ );
  scratch_.xgrads .resize   ( rank_,  nodeCount_ );
  scratch_.xgrads2.resize   ( rank2_, nodeCount_ );

  if ( xshape_ == sshape_ )
  {
    scratch_.sfuncs .ref    ( scratch_.xfuncs  );
    scratch_.sgrads .ref    ( scratch_.xgrads  );
    scratch_.sgrads2.ref    ( scratch_.xgrads2 );

    connect ( xshape_->changeEvent, this, &Self::shapeChanged_ );
  }
  else
  {
    scratch_.sfuncs .resize (         funcCount_ );
    scratch_.sgrads .resize ( rank_,  funcCount_ );
    scratch_.sgrads2.resize ( rank2_, funcCount_ );

    connect ( xshape_->changeEvent, this, &Self::xshapeChanged_ );
    connect ( sshape_->changeEvent, this, &Self::sshapeChanged_ );
  }
}


//-----------------------------------------------------------------------
//   shapeChanged_
//-----------------------------------------------------------------------


void ParametricShape::shapeChanged_ ()
{
  const idx_t  bndCount = bpoints_.size ();

  ipoints_ .updated = U_NONE_;
  vertices_.updated = U_NONE_;

  for ( idx_t ibnd = 0; ibnd < bndCount; ibnd++ )
  {
    bpoints_[ibnd].updated = U_NONE_;
  }

  changeEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   xshapeChanged_
//-----------------------------------------------------------------------


void ParametricShape::xshapeChanged_ ()
{
  const idx_t  bndCount = bpoints_.size ();

  ipoints_ .updated &= ~U_XGRADS2_;
  vertices_.updated &= ~U_XGRADS2_;

  for ( idx_t ibnd = 0; ibnd < bndCount; ibnd++ )
  {
    bpoints_[ibnd].updated &= ~U_XGRADS2_;
  }

  changeEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   sshapeChanged_
//-----------------------------------------------------------------------


void ParametricShape::sshapeChanged_ ()
{
  const idx_t  bndCount = bpoints_.size ();

  ipoints_ .updated &= ~U_SGRADS2_;
  vertices_.updated &= ~U_SGRADS2_;

  for ( idx_t ibnd = 0; ibnd < bndCount; ibnd++ )
  {
    bpoints_[ibnd].updated &= ~U_SGRADS2_;
  }

  changeEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   doUpdateFuncs_
//-----------------------------------------------------------------------


void ParametricShape::doUpdateFuncs_ ( PointSet_& points ) const
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


void ParametricShape::doUpdateGrads_ ( PointSet_& points ) const
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


//-----------------------------------------------------------------------
//   doUpdateGrads2_
//-----------------------------------------------------------------------


void ParametricShape::doUpdateGrads2_ ( PointSet_& points ) const
{
  const idx_t  pcount = points.coords.size(1);

  // Allocate memory if needed.

  if ( points.xgrads2.size(0) == 0 )
  {
    points.xfuncs .resize (         nodeCount_, pcount );
    points.xgrads .resize ( rank_,  nodeCount_, pcount );
    points.xgrads2.resize ( rank2_, nodeCount_, pcount );

    if ( sshape_ == xshape_ )
    {
      points.sfuncs .ref ( points.xfuncs );
      points.sgrads .ref ( points.xgrads );
      points.sgrads2.ref ( points.xgrads2 );
    }
    else
    {
      points.sfuncs .resize (         funcCount_, pcount );
      points.sgrads .resize ( rank_,  funcCount_, pcount );
      points.sgrads2.resize ( rank2_, funcCount_, pcount );
    }
  }

  if ( (points.updated & U_XGRADS2_) != U_XGRADS2_ )
  {
    for ( idx_t ip = 0; ip < pcount; ip++ )
    {
      xshape_->evalShapeGradGrads ( points.xfuncs [ip],
                                    points.xgrads [ip],
                                    points.xgrads2[ip],
                                    points.coords [ip] );
    }

    points.updated |= U_XGRADS2_;

    if ( sshape_ == xshape_ )
    {
      points.updated |= U_SGRADS2_;
    }
  }

  if ( (points.updated & U_SGRADS2_) != U_SGRADS2_ )
  {
    for ( idx_t ip = 0; ip < pcount; ip++ )
    {
      sshape_->evalShapeGradGrads ( points.sfuncs [ip],
                                    points.sgrads [ip],
                                    points.sgrads2[ip],
                                    points.coords [ip] );
    }

    points.updated |= U_SGRADS2_;
  }
}


JIVE_END_PACKAGE( geom )
