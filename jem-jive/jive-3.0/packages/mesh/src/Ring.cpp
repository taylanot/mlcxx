
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
#include <jive/mesh/Ring.h>


JEM_DEFINE_CLASS( jive::mesh::Ring );


JIVE_BEGIN_PACKAGE( mesh )


using jem::newInstance;


//=======================================================================
//   class Ring
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  Ring::TYPE_NAME = "Ring";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


Ring::Ring ( const String& name ) :

  Super   ( name ),
  xnode_  ( 2 )

{
  JIVE_INIT_GROUP_NAME ( innerEdge );
  JIVE_INIT_GROUP_NAME ( outerEdge );

  stripMesher_ =

    newInstance<StripMesher> ( StripMesher::WRAP_AROUND |
                               StripMesher::NUMBER_CLOCKWISE );

  elemType_    = TRIANGLE3;
  innerCount_  = 8;
  innerRadius_ = 1.0;
  outerRadius_ = 2.0;
}


Ring::~Ring ()
{}


//-----------------------------------------------------------------------
//   rank
//-----------------------------------------------------------------------


idx_t Ring::rank () const
{
  return 2;
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void Ring::configure ( const Properties& props )
{
  using jem::maxOf;
  using jem::minOf;

  if ( props.contains( myName_ ) )
  {
    Properties  myProps = props.findProps ( myName_ );

    double  r1 = innerRadius_;
    double  r2 = outerRadius_;


    ElemTypes::configure ( elemType_, myProps,
                           ElemTypes::TRIANGLE3,
                           ElemTypes::TRIANGLE6 );

    myProps.find ( innerCount_,
                   PropNames::INNER_NODES,
                   4, maxOf( innerCount_ ) );

    myProps.find ( r1,
                   PropNames::INNER_RADIUS,
                   minOf( r1 ), maxOf( r1 ) );

    myProps.find ( r2,
                   PropNames::OUTER_RADIUS,
                   r1 + minOf( r2 ), maxOf( r2 ) );

    innerRadius_ = r1;
    outerRadius_ = r2;

    if ( myProps.contains( PropNames::GROUP_NAMES ) )
    {
      Properties  gnProps =

        myProps.findProps ( PropNames::GROUP_NAMES );

      JIVE_FIND_GROUP_NAME ( innerEdge, gnProps );
      JIVE_FIND_GROUP_NAME ( outerEdge, gnProps );
    }
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void Ring::getConfig ( const Properties& conf ) const
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

  myConf.set ( PropNames::ELEM_TYPE,    typeName     );
  myConf.set ( PropNames::INNER_NODES,  innerCount_  );
  myConf.set ( PropNames::INNER_RADIUS, innerRadius_ );
  myConf.set ( PropNames::OUTER_RADIUS, outerRadius_ );

  JIVE_PUT_GROUP_NAME ( gnConf, innerEdge );
  JIVE_PUT_GROUP_NAME ( gnConf, outerEdge );
}


//-----------------------------------------------------------------------
//    makeMesh
//-----------------------------------------------------------------------


void Ring::makeMesh

  ( const XMesh&   mesh,
    const Params&  params )

{
  using jem::min;
  using jem::max;

  JEM_PRECHECK ( mesh && params.isValid() );

  const double  twoPi  = 2.0 * PI;
  const double  esize  = params.elemSize;
  const double  rmax   = params.scale * outerRadius_;

  Vector&       xinner = xinner_;
  Vector&       xouter = xouter_;

  IdxVector&    iinner = iinner_;
  IdxVector&    iouter = iouter_;

  Vector        xnode  = xnode_;

  double        t, dt;
  double        r, dr;
  double        a;

  idx_t         imid;
  idx_t         i, j, k, m, n, ne;


  if ( elemType_ == TRIANGLE3 )
  {
    ne = 1;
  }
  else
  {
    ne = 2;
  }

  r = params.scale * innerRadius_;
  n = (idx_t) (PI * (rmax * rmax - r * r) / (esize * esize));
  n = max ( n, 32_idx );

  mesh.nodes.reserve ( mesh.nodes.size() + n * ne );
  mesh.elems.reserve ( mesh.elems.size() + n );

  n = ne * max ( 1_idx, (idx_t) (twoPi * rmax / esize) );

  if ( n > xinner.size() )
  {
    xinner.resize ( n );
    xouter.resize ( n );
    iinner.resize ( n );
    iouter.resize ( n );
  }

  a  = twoPi * r;
  dr = min ( esize, a / 4.0, rmax - r );
  n  = segmentCount_ ( a, dr );
  n  = max ( n, innerCount_ );
  dr =     a / (double) n;
  dt = twoPi / (double) n;

  iinner.resize ( n * ne );
  xinner.resize ( n * ne);

  for ( j = 0; j < n; j++ )
  {
    t = (double) j * dt;
    k =          j * ne;

    xnode [0] = r * std::cos ( t );
    xnode [1] = r * std::sin ( t );
    xinner[k] = t;
    iinner[k] = mesh.nodes.addNode ( xnode );

    if ( elemType_ == TRIANGLE6 )
    {
      t += 0.5 * dt;
      k += 1;

      xnode [0] = r * std::cos ( t );
      xnode [1] = r * std::sin ( t );
      xinner[k] = t;
      iinner[k] = mesh.nodes.addNode ( xnode );
    }
  }

  if ( params.nodeGroups )
  {
    addGroup ( params, groupNames.innerEdge, iinner, mesh.nodes );
  }

  imid = -1;

  while ( (r + 3.0 * dr) < rmax )
  {
    if ( elemType_ == TRIANGLE6 )
    {
      xnode[0] = r + 0.5 * dr;
      xnode[1] = 0.0;
      imid     = mesh.nodes.addNode ( xnode );
    }

    r  += dr;
    a   = twoPi * r;
    n   = segmentCount_ ( a, dr );
    dt  = twoPi / (double) n;
    dr  = min ( esize, 1.2 * dr );

    iouter.resize ( n * ne );
    xouter.resize ( n * ne );

    for ( j = 0; j < n; j++ )
    {
      t = (double) j * dt;
      k =          j * ne;

      xnode [0] = r * std::cos ( t );
      xnode [1] = r * std::sin ( t );
      xouter[k] = t;
      iouter[k] = mesh.nodes.addNode ( xnode );

      if ( elemType_ == TRIANGLE6 )
      {
        t += 0.5 * dt;
        k += 1;

        xnode [0] = r * std::cos ( t );
        xnode [1] = r * std::sin ( t );
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

    xinner.swap ( xouter );
    iinner.swap ( iouter );
  }

  m  = (idx_t) ((rmax - r) / dr);
  m  = max ( m, 1_idx );
  dr = (rmax - r) / (double) m;

  for ( i = 1; i <= m; i++ )
  {
    if ( elemType_ == TRIANGLE6 )
    {
      xnode[0] = r + 0.5 * dr;
      xnode[1] = 0.0;
      imid     = mesh.nodes.addNode ( xnode );
    }

    r += dr;
    a  = twoPi * r;
    n  = segmentCount_ ( a, dr );
    dt = twoPi / (double) n;

    iouter.resize ( n * ne );
    xouter.resize ( n * ne );

    for ( j = 0; j < n; j++ )
    {
      t = (double) j * dt;
      k =          j * ne;

      xnode [0] = r * std::cos ( t );
      xnode [1] = r * std::sin ( t );
      xouter[k] = t;
      iouter[k] = mesh.nodes.addNode ( xnode );

      if ( elemType_ == TRIANGLE6 )
      {
        t += 0.5 * dt;
        k += 1;

        xnode [0] = r * std::cos ( t );
        xnode [1] = r * std::sin ( t );
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

    xinner.swap ( xouter );
    iinner.swap ( iouter );
  }

  if ( params.nodeGroups )
  {
    addGroup ( params, groupNames.outerEdge, iinner, mesh.nodes );
  }
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void Ring::declare ()
{
  GeometryFactory::declare ( TYPE_NAME,  & makeNew );
  GeometryFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Geometry> Ring::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  return newInstance<Self> ( name );
}


//-----------------------------------------------------------------------
//   segmentCount_
//-----------------------------------------------------------------------


inline idx_t Ring::segmentCount_ ( double a, double dr )
{
  idx_t  n = jem::max ( 4_idx, (idx_t) (a / dr) );
  idx_t  k = n % 4;

  if ( k > 0 )
  {
    n += (4 - k);
  }

  return n;
}


JIVE_END_PACKAGE( mesh )
