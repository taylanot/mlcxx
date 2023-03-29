
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
#include <jem/base/ClassTemplate.h>
#include <jive/util/GroupConverter.h>
#include <jive/geom/ShapeTable.h>
#include <jive/fem/BndGroupConverter.h>


JEM_DEFINE_CLASS( jive::fem::BndGroupConverter );


JIVE_BEGIN_PACKAGE( fem )


using jem::newInstance;


//=======================================================================
//   class BndGroupConverter
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


BndGroupConverter::BndGroupConverter

  ( const BoundarySet&      bounds,
    const Ref<ShapeTable>&  shapes,
    const Properties&       globdat,
    const String&           context ) :

    bounds_ ( bounds ),
    shapes_ ( shapes )

{
  JEM_PRECHECK ( bounds && shapes );

  ElementSet  elems = bounds.getElements ();
  NodeSet     nodes = elems .getNodes    ();

  conv_[ELEM_NODE_] =

    newInstance<Converter_> ( elems.getData (),
                              nodes.getData (),
                              globdat,
                              context );

  conv_[BND_NODE_]  =

    newInstance<Converter_> ( bounds.getData (),
                              nodes .getData (),
                              globdat,
                              context );

  conv_[BND_ELEM_]  =

    newInstance<Converter_> ( bounds.getData (),
                              elems .getData (),
                              globdat,
                              context );

}


BndGroupConverter::~BndGroupConverter ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


void BndGroupConverter::init

  ( const Properties&  conf,
    const Properties&  props )

{
  for ( int i = 0; i < 3; i++ )
  {
    conv_[i]->init ( conf, props );
  }
}


//-----------------------------------------------------------------------
//   isDone
//-----------------------------------------------------------------------


bool BndGroupConverter::isDone () const
{
  for ( int i = 0; i < 3; i++ )
  {
    if ( ! conv_[i]->isDone() )
    {
      return false;
    }
  }

  return true;
}


//-----------------------------------------------------------------------
//   convert
//-----------------------------------------------------------------------


idx_t BndGroupConverter::convert ()
{
  using jive::util::Topology;
  using jive::util::TopoVector;


  idx_t  n = 0;

  if ( ! conv_[ELEM_NODE_]->isDone() )
  {
    ElementSet  elems = bounds_.getElements ();
    NodeSet     nodes = elems  .getNodes    ();

    Topology    enMap = elems.toMatrix  ();
    Topology    neMap = enMap.transpose ();


    n += conv_[ELEM_NODE_]->convert ( enMap, neMap );
  }

  if ( ! conv_[BND_ELEM_]->isDone() )
  {
    Topology  ebMap = bounds_.getBoundsPerElement ();
    Topology  beMap = ebMap  .transpose           ();


    n += conv_[BND_ELEM_]->convert ( beMap, ebMap );
  }

  // The most expensive conversion is done last.

  if ( ! conv_[BND_NODE_]->isDone() )
  {
    ElementSet  elems     = bounds_.getElements ();
    NodeSet     nodes     = elems  .getNodes    ();

    IdxVector   topoMap   ( elems.size() );

    TopoVector  elemTopos = shapes_->getShapeTopos ( topoMap );

    Topology    bnMap     =

      bounds_.getNodesPerBoundary ( topoMap, elemTopos );

    Topology    nbMap     = bnMap.transpose ();


    n += conv_[BND_NODE_]->convert ( bnMap, nbMap );
  }


  return n;
}


//-----------------------------------------------------------------------
//   finish
//-----------------------------------------------------------------------


void BndGroupConverter::finish ()
{
  for ( int i = 0; i < 3; i++ )
  {
    conv_[i]->finish ();
  }
}


JIVE_END_PACKAGE( fem )
