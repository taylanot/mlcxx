
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
#include <jive/mesh/Circle.h>


JEM_DEFINE_CLASS( jive::mesh::Circle );


JIVE_BEGIN_PACKAGE( mesh )


using jem::newInstance;


//=======================================================================
//   class Circle
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  Circle::TYPE_NAME = "Circle";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


Circle::Circle ( const String& name ) :

  Super  ( name ),
  xnode_ ( 2 )

{
  JIVE_INIT_GROUP_NAME ( edge   );
  JIVE_INIT_GROUP_NAME ( center );

  stripMesher_ =

    newInstance<StripMesher> ( StripMesher::WRAP_AROUND |
                               StripMesher::NUMBER_CLOCKWISE );

  radius_   = 1.0;
  elemType_ = TRIANGLE3;
}


Circle::~Circle ()
{}


//-----------------------------------------------------------------------
//   rank
//-----------------------------------------------------------------------


idx_t Circle::rank () const
{
  return 2;
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void Circle::configure ( const Properties& props )
{
  using jem::minOf;
  using jem::maxOf;

  if ( props.contains( myName_ ) )
  {
    Properties  myProps = props.findProps ( myName_ );

    myProps.find ( radius_,
                   PropNames::RADIUS,
                   minOf ( radius_ ),
                   maxOf ( radius_ ) );

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


void Circle::getConfig ( const Properties& conf ) const
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

  myConf.set ( PropNames::RADIUS,    radius_  );
  myConf.set ( PropNames::ELEM_TYPE, typeName );

  JIVE_PUT_GROUP_NAME ( gnConf, edge   );
  JIVE_PUT_GROUP_NAME ( gnConf, center );
}


//-----------------------------------------------------------------------
//    makeMesh
//-----------------------------------------------------------------------


void Circle::makeMesh

  ( const XMesh&   mesh,
    const Params&  params )

{
  using jem::max;

  JEM_PRECHECK ( mesh && params.isValid() );

  const double  radius = params.scale * radius_;

  Vector&       xouter = xouter_;
  Vector&       xinner = xinner_;

  IdxVector&    iouter = iouter_;
  IdxVector&    iinner = iinner_;

  Vector        xnode  = xnode_;

  double        t, dt;
  double        r, dr;

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

  stripCount = max ( 1_idx, (idx_t)
                     ((radius / params.elemSize) + 0.5) );
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

  if ( params.nodeGroups )
  {
    addGroup ( params, groupNames.center, iinner, mesh.nodes );
  }

  r    = 0.0;
  dr   = radius / (double) stripCount;
  imid = -1;

  for ( i = 0; i < stripCount; i++ )
  {
    if ( elemType_ == TRIANGLE6 )
    {
      xnode[0] = r + 0.5 * dr;
      xnode[1] = 0.0;
      imid     = mesh.nodes.addNode ( xnode );
    }

    n = 8 * (i + 1);

    xouter.resize ( n * ne );
    iouter.resize ( n * ne );

    r  = r + dr;
    dt = 2.0 * PI / (double) n;

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

    iinner.swap ( iouter );
    xinner.swap ( xouter );
  }

  if ( params.nodeGroups )
  {
    addGroup ( params, groupNames.edge, iinner, mesh.nodes );
  }
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void Circle::declare ()
{
  GeometryFactory::declare ( TYPE_NAME,  & makeNew );
  GeometryFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Geometry> Circle::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  return newInstance<Self> ( name );
}


JIVE_END_PACKAGE( mesh )
