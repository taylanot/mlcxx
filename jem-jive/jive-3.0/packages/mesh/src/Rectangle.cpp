
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
#include <jem/util/Properties.h>
#include <jive/mesh/groupnames.h>
#include <jive/mesh/Mesh.h>
#include <jive/mesh/Names.h>
#include <jive/mesh/ElementTypes.h>
#include <jive/mesh/GeometryFactory.h>
#include <jive/mesh/Rectangle.h>


JEM_DEFINE_CLASS( jive::mesh::Rectangle );


JIVE_BEGIN_PACKAGE( mesh )


//=======================================================================
//   class Rectangle
//=======================================================================

//-----------------------------------------------------------------------
//  static data
//-----------------------------------------------------------------------


const char*  Rectangle::TYPE_NAME = "Rectangle";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


Rectangle::Rectangle ( const String& name ) :

  Super  ( name ),
  xnode_ ( 2 )

{
  rsize_[0] = 1.0;
  rsize_[1] = 1.0;
  elemType_ = QUAD4;

  JIVE_INIT_GROUP_NAME ( leftEdge  );
  JIVE_INIT_GROUP_NAME ( rightEdge );
  JIVE_INIT_GROUP_NAME ( lowerEdge );
  JIVE_INIT_GROUP_NAME ( upperEdge );
}


Rectangle::~Rectangle ()
{}


//-----------------------------------------------------------------------
//   rank
//-----------------------------------------------------------------------


idx_t Rectangle::rank () const
{
  return 2;
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void Rectangle::configure ( const Properties& props )
{
  using jem::minOf;
  using jem::maxOf;

  if ( props.contains( myName_ ) )
  {
    Properties  myProps = props.findProps ( myName_ );

    Vector      rsize ( 2 );


    rsize[0] = rsize_[0];
    rsize[1] = rsize_[1];

    myProps.find ( rsize,
                   PropNames::SIZE,
                   minOf<double> (),
                   maxOf<double> () );

    if ( rsize.size() != 2 )
    {
      myProps.propertyError (
        PropNames::SIZE,
        "invalid array length; should be two"
      );
    }

    rsize_[0] = rsize[0];
    rsize_[1] = rsize[1];

    ElemTypes::configure ( elemType_, myProps,
                           ElemTypes::QUAD4,
                           ElemTypes::QUAD8,
                           ElemTypes::QUAD12 );

    if ( myProps.contains( PropNames::GROUP_NAMES ) )
    {
      Properties  gnProps =

        myProps.findProps ( PropNames::GROUP_NAMES );

      JIVE_FIND_GROUP_NAME ( leftEdge,  gnProps );
      JIVE_FIND_GROUP_NAME ( rightEdge, gnProps );
      JIVE_FIND_GROUP_NAME ( lowerEdge, gnProps );
      JIVE_FIND_GROUP_NAME ( upperEdge, gnProps );
    }
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void Rectangle::getConfig ( const Properties& conf ) const
{
  Properties  myConf = conf  .makeProps ( myName_ );
  Properties  gnConf = myConf.makeProps ( PropNames::GROUP_NAMES );

  Vector      rsize ( 2 );

  String      typeName;


  rsize[0] = rsize_[0];
  rsize[1] = rsize_[1];

  if      ( elemType_ == QUAD4 )
  {
    typeName = ElemTypes::QUAD4;
  }
  else if ( elemType_ == QUAD8 )
  {
    typeName = ElemTypes::QUAD8;
  }
  else
  {
    typeName = ElemTypes::QUAD12;
  }

  myConf.set ( PropNames::SIZE,      rsize );
  myConf.set ( PropNames::ELEM_TYPE, typeName );

  JIVE_PUT_GROUP_NAME ( gnConf, leftEdge  );
  JIVE_PUT_GROUP_NAME ( gnConf, rightEdge );
  JIVE_PUT_GROUP_NAME ( gnConf, lowerEdge );
  JIVE_PUT_GROUP_NAME ( gnConf, upperEdge );
}


//-----------------------------------------------------------------------
//    makeMesh
//-----------------------------------------------------------------------


void Rectangle::makeMesh

  ( const XMesh&   mesh,
    const Params&  params )

{
  using jem::max;
  using jem::min;

  JEM_PRECHECK ( mesh && params.isValid() );

  const double  xsize = rsize_[0] * params.scale;
  const double  ysize = rsize_[1] * params.scale;

  double        delta;
  double        dx, dy;

  idx_t         m, n;


  delta = 2.0 * min ( xsize, ysize );
  delta =       min ( delta, params.elemSize );

  m  = max ( 2_idx, (idx_t) (xsize / delta + 1.5) );
  n  = max ( 2_idx, (idx_t) (ysize / delta + 1.5) );

  dx = xsize / (double) (m - 1);
  dy = ysize / (double) (n - 1);

  mesh.elems.reserve ( mesh.elems.size() + m * n );

  if      ( elemType_ == QUAD4 )
  {
    makeMesh4_  ( mesh, m, n, dx, dy );
  }
  else if ( elemType_ == QUAD8 )
  {
    makeMesh8_  ( mesh, m, n, dx, dy );
  }
  else
  {
    makeMesh12_ ( mesh, m, n, dx, dy );
  }

  if ( params.nodeGroups )
  {
    IdxMatrix&  gridNodes = gridNodes_;

    m = gridNodes.size(0) - 1;
    n = gridNodes.size(1) - 1;

    addGroup ( params,
               groupNames.leftEdge,
               gridNodes(0,ALL),
               mesh.nodes );

    addGroup ( params,
               groupNames.rightEdge,
               gridNodes(m,ALL),
               mesh.nodes );

    addGroup ( params,
               groupNames.lowerEdge,
               gridNodes[0],
               mesh.nodes );

    addGroup ( params,
               groupNames.upperEdge,
               gridNodes[n],
               mesh.nodes );
  }
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void Rectangle::declare ()
{
  GeometryFactory::declare ( TYPE_NAME,  & makeNew );
  GeometryFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Geometry> Rectangle::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  return jem::newInstance<Self> ( name );
}


//-----------------------------------------------------------------------
//   makeMesh4_
//-----------------------------------------------------------------------


void Rectangle::makeMesh4_

  ( const XMesh&  mesh,
    idx_t         m,
    idx_t         n,
    double        dx,
    double        dy )

{
  Vector&     xnode     = xnode_;
  IdxMatrix&  gridNodes = gridNodes_;
  IdxVector&  elemNodes = elemNodes_;


  mesh.nodes.reserve ( mesh.nodes.size() + m * n );
  gridNodes .resize  ( m, n );
  elemNodes .resize  ( 4 );

  // Choose a bandwidth-minimizing numbering scheme.

  if ( m < n )
  {
    for ( idx_t j = 0; j < n; j++ )
    {
      xnode[1] = (double) j * dy;

      for ( idx_t i = 0; i < m; i++ )
      {
        xnode[0]       = (double) i * dx;
        gridNodes(i,j) = mesh.nodes.addNode ( xnode );
      }
    }
  }
  else
  {
    for ( idx_t i = 0; i < m; i++ )
    {
      xnode[0] = (double) i * dx;

      for ( idx_t j = 0; j < n; j++ )
      {
        xnode[1]       = (double) j * dy;
        gridNodes(i,j) = mesh.nodes.addNode ( xnode );
      }
    }
  }

  for ( idx_t j = 0; j < n - 1; j++ )
  {
    for ( idx_t i = 0; i < m - 1; i++ )
    {
      elemNodes[0] = gridNodes(i + 0,j + 0);
      elemNodes[1] = gridNodes(i + 1,j + 0);
      elemNodes[2] = gridNodes(i + 1,j + 1);
      elemNodes[3] = gridNodes(i + 0,j + 1);

      mesh.elems.addElement ( elemNodes );
    }
  }
}


//-----------------------------------------------------------------------
//   makeMesh8_
//-----------------------------------------------------------------------


void Rectangle::makeMesh8_

  ( const XMesh&  mesh,
    idx_t         m,
    idx_t         n,
    double        dx,
    double        dy )

{
  const idx_t  m2        = m * 2 - 1;
  const idx_t  n2        = n * 2 - 1;

  Vector&      xnode     = xnode_;
  IdxMatrix&   gridNodes = gridNodes_;
  IdxVector&   elemNodes = elemNodes_;


  mesh.nodes.reserve ( mesh.nodes.size() + (3 * m * n - m - n) );
  gridNodes .resize  ( m2, n2 );
  elemNodes .resize  ( 8 );

  gridNodes = -1;

  // Choose a bandwidth-minimizing numbering scheme.

  if ( m < n )
  {
    for ( idx_t j = 0; j < n; j++ )
    {
      idx_t  jj = j * 2;

      xnode[1] = (double) j * dy;

      for ( idx_t i = 0; i < m; i++ )
      {
        idx_t  ii = i * 2;

        xnode[0]         = (double) i * dx;
        gridNodes(ii,jj) = mesh.nodes.addNode ( xnode );

        ii++;

        if ( ii < m2 )
        {
          xnode[0]        += 0.5 * dx;
          gridNodes(ii,jj) = mesh.nodes.addNode ( xnode );
        }
      }

      jj++;

      if ( jj < n2 )
      {
        xnode[1] += 0.5 * dy;

        for ( idx_t i = 0; i < m; i++ )
        {
          idx_t  ii = i * 2;

          xnode[0]         = (double) i * dx;
          gridNodes(ii,jj) = mesh.nodes.addNode ( xnode );
        }
      }
    }
  }
  else
  {
    for ( idx_t i = 0; i < m; i++ )
    {
      idx_t  ii = i * 2;

      xnode[0] = (double) i * dx;

      for ( idx_t j = 0; j < n; j++ )
      {
        idx_t  jj = j * 2;

        xnode[1]         = (double) j * dy;
        gridNodes(ii,jj) = mesh.nodes.addNode ( xnode );

        jj++;

        if ( jj < n2 )
        {
          xnode[1]        += 0.5 * dy;
          gridNodes(ii,jj) = mesh.nodes.addNode ( xnode );
        }
      }

      ii++;

      if ( ii < m2 )
      {
        xnode[0] += 0.5 * dx;

        for ( idx_t j = 0; j < n; j++ )
        {
          idx_t  jj = j * 2;

          xnode[1]         = (double) j * dy;
          gridNodes(ii,jj) = mesh.nodes.addNode ( xnode );
        }
      }
    }
  }

  for ( idx_t j = 0; j < n - 1; j++ )
  {
    idx_t  jj = j * 2;

    for ( idx_t i = 0; i < m - 1; i++ )
    {
      idx_t  ii = i * 2;

      elemNodes[0] = gridNodes(ii + 0,jj + 0);
      elemNodes[1] = gridNodes(ii + 1,jj + 0);
      elemNodes[2] = gridNodes(ii + 2,jj + 0);
      elemNodes[3] = gridNodes(ii + 2,jj + 1);
      elemNodes[4] = gridNodes(ii + 2,jj + 2);
      elemNodes[5] = gridNodes(ii + 1,jj + 2);
      elemNodes[6] = gridNodes(ii + 0,jj + 2);
      elemNodes[7] = gridNodes(ii + 0,jj + 1);

      mesh.elems.addElement ( elemNodes );
    }
  }
}


//-----------------------------------------------------------------------
//   makeMesh12_
//-----------------------------------------------------------------------


void Rectangle::makeMesh12_

  ( const XMesh&  mesh,
    idx_t         m,
    idx_t         n,
    double        dx,
    double        dy )

{
  const idx_t  m3        = m * 3 - 2;
  const idx_t  n3        = n * 3 - 2;

  Vector&      xnode     = xnode_;
  IdxMatrix&   gridNodes = gridNodes_;
  IdxVector&   elemNodes = elemNodes_;


  mesh.nodes.reserve ( mesh.nodes.size() +
                       (5 * m * n - 2 * m - 2 * n) );
  gridNodes .resize  ( m3, n3 );
  elemNodes .resize  ( 12 );

  gridNodes = -1;

  // Choose a bandwidth-minimizing numbering scheme.

  if ( m < n )
  {
    for ( idx_t j = 0; j < n; j++ )
    {
      idx_t  jj = j * 3;

      xnode[1] = (double) j * dy;

      for ( idx_t i = 0; i < m; i++ )
      {
        idx_t  ii = i * 3;

        xnode[0]         = (double) i * dx;
        gridNodes(ii,jj) = mesh.nodes.addNode ( xnode );

        ii++;

        if ( ii < m3 )
        {
          xnode[0]        += dx / 3.0;
          gridNodes(ii,jj) = mesh.nodes.addNode ( xnode );

          ii++;

          xnode[0]        += dx / 3.0;
          gridNodes(ii,jj) = mesh.nodes.addNode ( xnode );
        }
      }

      jj++;

      if ( jj < n3 )
      {
        xnode[1] += dy / 3.0;

        for ( idx_t i = 0; i < m; i++ )
        {
          idx_t  ii = i * 3;

          xnode[0]         = (double) i * dx;
          gridNodes(ii,jj) = mesh.nodes.addNode ( xnode );
        }

        jj++;

        xnode[1] += dy / 3.0;

        for ( idx_t i = 0; i < m; i++ )
        {
          idx_t  ii = i * 3;

          xnode[0]         = (double) i * dx;
          gridNodes(ii,jj) = mesh.nodes.addNode ( xnode );
        }
      }
    }
  }
  else
  {
    for ( idx_t i = 0; i < m; i++ )
    {
      idx_t  ii = i * 3;

      xnode[0] = (double) i * dx;

      for ( idx_t j = 0; j < n; j++ )
      {
        idx_t  jj = j * 3;

        xnode[1]         = (double) j * dy;
        gridNodes(ii,jj) = mesh.nodes.addNode ( xnode );

        jj++;

        if ( jj < n3 )
        {
          xnode[1]        += dy / 3.0;
          gridNodes(ii,jj) = mesh.nodes.addNode ( xnode );

          jj++;

          xnode[1]        += dy / 3.0;
          gridNodes(ii,jj) = mesh.nodes.addNode ( xnode );
        }
      }

      ii++;

      if ( ii < m3 )
      {
        xnode[0] += dx / 3.0;

        for ( idx_t j = 0; j < n; j++ )
        {
          idx_t  jj = j * 3;

          xnode[1]         = (double) j * dy;
          gridNodes(ii,jj) = mesh.nodes.addNode ( xnode );
        }

        ii++;

        xnode[0] += dx / 3.0;

        for ( idx_t j = 0; j < n; j++ )
        {
          idx_t  jj = j * 3;

          xnode[1]         = (double) j * dy;
          gridNodes(ii,jj) = mesh.nodes.addNode ( xnode );
        }
      }
    }
  }

  for ( idx_t j = 0; j < n - 1; j++ )
  {
    idx_t  jj = j * 3;

    for ( idx_t i = 0; i < m - 1; i++ )
    {
      idx_t  ii = i * 3;

      elemNodes[ 0] = gridNodes(ii + 0,jj + 0);
      elemNodes[ 1] = gridNodes(ii + 1,jj + 0);
      elemNodes[ 2] = gridNodes(ii + 2,jj + 0);
      elemNodes[ 3] = gridNodes(ii + 3,jj + 0);
      elemNodes[ 4] = gridNodes(ii + 3,jj + 1);
      elemNodes[ 5] = gridNodes(ii + 3,jj + 2);
      elemNodes[ 6] = gridNodes(ii + 3,jj + 3);
      elemNodes[ 7] = gridNodes(ii + 2,jj + 3);
      elemNodes[ 8] = gridNodes(ii + 1,jj + 3);
      elemNodes[ 9] = gridNodes(ii + 0,jj + 3);
      elemNodes[10] = gridNodes(ii + 0,jj + 2);
      elemNodes[11] = gridNodes(ii + 0,jj + 1);

      mesh.elems.addElement ( elemNodes );
    }
  }
}

JIVE_END_PACKAGE( mesh )
