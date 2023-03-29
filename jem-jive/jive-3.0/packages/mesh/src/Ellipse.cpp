
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
#include <jem/base/assert.h>
#include <jem/base/limits.h>
#include <jem/base/ClassTemplate.h>
#include <jem/util/Properties.h>
#include <jive/mesh/groupnames.h>
#include <jive/mesh/Mesh.h>
#include <jive/mesh/Names.h>
#include <jive/mesh/StripMesher.h>
#include <jive/mesh/ElementTypes.h>
#include <jive/mesh/GeometryFactory.h>
#include <jive/mesh/Ellipse.h>


JEM_DEFINE_CLASS( jive::mesh::Ellipse );


JIVE_BEGIN_PACKAGE( mesh )


//=======================================================================
//   class Ellipse
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  Ellipse::TYPE_NAME = "Ellipse";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


Ellipse::Ellipse ( const String& name ) :

  Super  ( name ),
  xnode_ ( 2 )

{
  JIVE_INIT_GROUP_NAME ( edge   );
  JIVE_INIT_GROUP_NAME ( center );

  stripMesher_ = jem::newInstance<StripMesher> ();

  xradius_     = 1.0;
  yradius_     = 1.0;
  elemType_    = TRIANGLE3;
}


Ellipse::~Ellipse ()
{}


//-----------------------------------------------------------------------
//   rank
//-----------------------------------------------------------------------


idx_t Ellipse::rank () const
{
  return 2;
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void Ellipse::configure ( const Properties& props )
{
  using jem::minOf;
  using jem::maxOf;

  if ( props.contains( myName_ ) )
  {
    Properties  myProps = props.findProps ( myName_ );

    myProps.find ( xradius_,
                   PropNames::XRADIUS,
                   minOf ( xradius_ ),
                   maxOf ( xradius_ ) );

    myProps.find ( yradius_,
                   PropNames::YRADIUS,
                   minOf ( yradius_ ),
                   maxOf ( yradius_ ) );

    ElemTypes::configure ( elemType_, myProps,
                           ElemTypes::TRIANGLE3,
                           ElemTypes::TRIANGLE6 );

    if ( myProps.contains( PropNames::GROUP_NAMES ) )
    {
      Properties  gnProps =

        myProps.findProps ( PropNames::GROUP_NAMES );

      JIVE_FIND_GROUP_NAME ( edge,   gnProps );
      JIVE_FIND_GROUP_NAME ( center, gnProps );
    }
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void Ellipse::getConfig ( const Properties& conf ) const
{
  Properties  myConf = conf  .makeProps ( myName_ );
  Properties  gnConf = myConf.makeProps ( PropNames::GROUP_NAMES );

  String      typeName;

  if ( elemType_ == TRIANGLE3 )
  {
    typeName = ElemTypes::TRIANGLE3;
  }
  else
  {
    typeName = ElemTypes::TRIANGLE6;
  }

  myConf.set ( PropNames::XRADIUS,   xradius_ );
  myConf.set ( PropNames::YRADIUS,   yradius_ );
  myConf.set ( PropNames::ELEM_TYPE, typeName );

  JIVE_PUT_GROUP_NAME ( gnConf, edge   );
  JIVE_PUT_GROUP_NAME ( gnConf, center );
}


//-----------------------------------------------------------------------
//    makeMesh
//-----------------------------------------------------------------------


void Ellipse::makeMesh

  ( const XMesh&   mesh,
    const Params&  params )

{
  JEM_PRECHECK ( mesh && params.isValid() );

  const double  r = xradius_ / yradius_;

  if ( r > 0.5 && r < 2.0 )
  {
    makeCircularMesh_  ( mesh, params );
  }
  else
  {
    makeElongatedMesh_ ( mesh, params );
  }
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void Ellipse::declare ()
{
  GeometryFactory::declare ( TYPE_NAME,  & makeNew );
  GeometryFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Geometry> Ellipse::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  return jem::newInstance<Self> ( name );
}


//-----------------------------------------------------------------------
//    makeCircularMesh_
//-----------------------------------------------------------------------


void Ellipse::makeCircularMesh_

  ( const XMesh&   mesh,
    const Params&  params )

{
  using jem::min;
  using jem::max;

  const double  xradius = xradius_ * params.scale;
  const double  yradius = yradius_ * params.scale;

  Vector&       xouter  = xouter_;
  IdxVector&    iouter  = iouter_;

  Vector&       xinner  = xinner_;
  IdxVector&    iinner  = iinner_;

  Vector        xnode   = xnode_;

  double        delta;
  double        t, dt;
  double        x, dx;
  double        y;

  idx_t         stripCount;
  idx_t         imid;
  idx_t         i, j, k, n, ne;


  if ( elemType_ == TRIANGLE3 )
  {
    ne = 1;
  }
  else
  {
    ne = 2;
  }

  delta      = 2.0 * min ( xradius, yradius );
  delta      =       min ( delta, params.elemSize );
  stripCount =       max ( 1_idx, (idx_t)
                           (xradius / delta + 0.5) );
  n          = 4 * stripCount * (stripCount - 1) + 1;

  mesh.nodes.reserve ( mesh.nodes.size() + n * ne );
  mesh.elems.reserve ( mesh.elems.size() + n );

  n *= ne;

  if ( n > xinner.size() )
  {
    xinner.resize ( n );
    xouter.resize ( n );
    iinner.resize ( n );
    iouter.resize ( n );
  }

  xinner.resize ( 1 );
  iinner.resize ( 1 );

  xnode [0] = 0.0;
  xnode [1] = 0.0;
  xinner[0] = 0.0;
  iinner[0] = mesh.nodes.addNode( xnode );

  if( params.nodeGroups )
  {
    addGroup ( params, groupNames.center, iinner, mesh.nodes );
  }

  stripMesher_->setOptions ( StripMesher::WRAP_AROUND |
                             StripMesher::NUMBER_CLOCKWISE );

  dx   = xradius / (double) stripCount;
  x    = 0.0;
  y    = 0.0;
  imid = -1;

  for ( i = 0; i < stripCount; i++ )
  {
    if ( elemType_ == TRIANGLE6 )
    {
      xnode[0] = x + 0.5 * dx;
      xnode[1] = 0.0;
      imid     = mesh.nodes.addNode ( xnode );
    }

    n = 8 * (i + 1);

    iouter.resize ( n * ne );
    xouter.resize ( n * ne );

    x  = x + dx;
    y  = x * yradius / xradius;
    dt = 2.0 * PI / (double) n;

    for ( j = 0; j < n; j++ )
    {
      t = (double) j * dt;
      k =          j * ne;

      xnode [0] = x * std::cos ( t );
      xnode [1] = y * std::sin ( t );
      iouter[k] = mesh.nodes.addNode ( xnode );
      xouter[k] = t;

      if ( elemType_ == TRIANGLE6 )
      {
        t += 0.5 * dt;
        k += 1;

        xnode [0] = x * std::cos ( t );
        xnode [1] = y * std::sin ( t );
        xouter[k] = t;
        iouter[k] = mesh.nodes.addNode ( xnode );
      }
    }

    if ( elemType_ == TRIANGLE3 )
    {
      stripMesher_->triangulate3 ( mesh.elems,
                                   xouter, iouter,
                                   xinner, iinner );
    }
    else
    {
      stripMesher_->triangulate6 ( mesh,
                                   imid, xouter, iouter,
                                   imid, xinner, iinner );
    }

    iinner.swap ( iouter );
    xinner.swap ( xouter );
  }

  if ( params.nodeGroups )
  {
    addGroup ( params, groupNames.edge, iinner, mesh.nodes );
  }
}


//-----------------------------------------------------------------------
//    makeElongatedMesh_
//-----------------------------------------------------------------------


void Ellipse::makeElongatedMesh_

  ( const XMesh&   mesh,
    const Params&  params )

{
  using jem::min;
  using jem::max;


  const double  xradius   = xradius_ * params.scale;
  const double  yradius   = yradius_ * params.scale;

  const bool    mkGroups  = (params.nodeGroups != nullptr);

  IdxVector&    edgeNodes = edgeNodes_;

  Vector&       xouter    = xouter_;
  IdxVector&    iouter    = iouter_;
  Vector&       xinner    = xinner_;
  IdxVector&    iinner    = iinner_;

  Vector        xnode     = xnode_;

  double        ra, rb;
  double        da, db;
  double        a,  b;
  double        r,  t;

  idx_t         stripCount;
  idx_t         adim,  bdim;
  idx_t         ileft, iright;
  idx_t         iedge;
  idx_t         i, j, k, n, ne;


  if ( elemType_ == TRIANGLE3 )
  {
    ne = 1;
  }
  else
  {
    ne = 2;
  }

  if ( xradius > yradius )
  {
    adim = 1;
    bdim = 0;
    ra   = yradius;
    rb   = xradius;

    stripMesher_->setOptions ( StripMesher::NUMBER_CLOCKWISE );
  }
  else
  {
    adim = 0;
    bdim = 1;
    ra   = xradius;
    rb   = yradius;

    stripMesher_->setOptions ( 0 );
  }

  stripCount = 2 * max ( 1_idx,
                         (idx_t) (rb / params.elemSize + 0.5) );
  n          = 2 * max ( 1_idx,
                         (idx_t) (ra / params.elemSize + 0.5) );

  mesh.nodes.reserve ( mesh.nodes.size() + n * stripCount * ne );
  mesh.elems.reserve ( mesh.elems.size() + n * stripCount );

  n *= ne;

  if ( n > xinner.size() )
  {
    xinner.resize ( n );
    xouter.resize ( n );
    iinner.resize ( n );
    iouter.resize ( n );
  }

  xinner.resize ( 1 );
  iinner.resize ( 1 );

  xnode[adim] =  0.0;
  xnode[bdim] = -rb;
  xinner[0]   =  0.0;
  iinner[0]   =  mesh.nodes.addNode ( xnode );

  db     =  2.0 * rb / (double) stripCount;
  ileft  = -1;
  iright = -1;
  iedge  =  0;

  if ( mkGroups )
  {
    edgeNodes.resize ( 2 * ne * stripCount );

    edgeNodes[iedge++] = iinner[0];
  }

  for ( i = 1; i < stripCount; i++ )
  {
    if ( elemType_ == TRIANGLE6 )
    {
      b = (double) i * db - rb - 0.5 * db;
      t = b / rb;
      r = ra * std::sqrt ( 1.0 - t * t );

      xnode[adim] = -r;
      xnode[bdim] =  b;
      ileft       =  mesh.nodes.addNode ( xnode );
      xnode[adim] =  r;
      iright      =  mesh.nodes.addNode ( xnode );
    }

    b  = (double) i * db - rb;
    t  = b / rb;
    r  = ra * std::sqrt ( 1.0 - t * t );
    n  = 2 * max ( 1_idx, (idx_t) (r / params.elemSize + 0.5) );
    da = 2.0 * r / (double) n;

    iouter.resize ( n * ne + 1 );
    xouter.resize ( n * ne + 1 );

    xnode[bdim] = b;

    for ( j = 0; j < n; j++ )
    {
      a = (double) j * da - r;
      k =          j * ne;

      xnode[adim] = a;
      xouter[k]   = a;
      iouter[k]   = mesh.nodes.addNode ( xnode );

      if ( elemType_ == TRIANGLE6 )
      {
        a += 0.5 * da;
        k += 1;

        xnode[adim] = a;
        xouter[k]   = a;
        iouter[k]   = mesh.nodes.addNode ( xnode );
      }
    }

    k = n * ne;

    xnode[adim] = r;
    xouter[k]   = r;
    iouter[k]   = mesh.nodes.addNode ( xnode );

    if ( elemType_ == TRIANGLE3 )
    {
      stripMesher_->triangulate3 ( mesh.elems,
                                   xouter, iouter,
                                   xinner, iinner );
    }
    else
    {
      stripMesher_->triangulate6 ( mesh,
                                   iright, xouter, iouter,
                                   ileft,  xinner, iinner );
    }

    if ( mkGroups )
    {
      if ( 2 * i == stripCount )
      {
        k = ne * n / 2;

        addGroup ( params,
                   groupNames.center,
                   iouter[slice(k,k + 1)],
                   mesh.nodes );
      }

      if ( elemType_ == TRIANGLE6 )
      {
        edgeNodes[iedge++] = ileft;
        edgeNodes[iedge++] = iright;
      }

      edgeNodes[iedge++] = iouter[0];
      edgeNodes[iedge++] = iouter[n * ne];
    }

    xouter.swap ( xinner );
    iouter.swap ( iinner );
  }

  if ( elemType_ == TRIANGLE6 )
  {
    b = rb - 0.5 * db;
    t = b / rb;
    r = ra * std::sqrt ( 1.0 - t * t );

    xnode[adim] = -r;
    xnode[bdim] =  b;
    ileft       =  mesh.nodes.addNode ( xnode );
    xnode[adim] =  r;
    iright      =  mesh.nodes.addNode ( xnode );
  }

  iouter.resize ( 1 );
  xouter.resize ( 1 );

  xnode[adim] = 0.0;
  xnode[bdim] = rb;
  xouter[0]   = 0.0;
  iouter[0]   = mesh.nodes.addNode ( xnode );

  if ( elemType_ == TRIANGLE3 )
  {
    stripMesher_->triangulate3 ( mesh.elems,
                                 xouter, iouter,
                                 xinner, iinner );
  }
  else
  {
    stripMesher_->triangulate6 ( mesh,
                                 iright, xouter, iouter,
                                 ileft,  xinner, iinner );
  }

  if ( mkGroups )
  {
    if ( elemType_ == TRIANGLE6 )
    {
      edgeNodes[iedge++] = ileft;
      edgeNodes[iedge++] = iright;
    }

    edgeNodes[iedge++] = iouter[0];

    addGroup ( params, groupNames.edge, edgeNodes, mesh.nodes );
  }
}


JIVE_END_PACKAGE( mesh )
