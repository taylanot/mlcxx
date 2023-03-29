
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


#include <jem/base/Array.h>
#include <jem/base/System.h>
#include <jem/base/ClassTemplate.h>
#include <jem/numeric/algebra/matmul.h>
#include <jem/util/Properties.h>
#include <jive/util/XTable.h>
#include <jive/util/Database.h>
#include <jive/util/XDofSpace.h>
#include <jive/algebra/MatrixBuilder.h>
#include <jive/model/Actions.h>
#include <jive/model/StateVector.h>
#include <jive/model/ModelFactory.h>
#include <jive/fem/NodeSet.h>
#include <jive/fem/ElementSet.h>
#include <jive/fem/BezierElement.h>
#include <jive/geom/StdSquare.h>
#include <jive/geom/StdBezierShape.h>
#include <jive/geom/ParametricArea.h>

#include "PoissonModel.h"


//=======================================================================
//   class PoissonModel
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  PoissonModel::TYPE_NAME = "Poisson";


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


PoissonModel::PoissonModel

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat ) :

    Model ( name )

{
  using jive::util::Database;
  using jive::geom::StdShape;
  using jive::geom::StdSquare;
  using jive::geom::StdSquare9B;
  using jive::geom::StdBezierShape;
  using jive::geom::ParametricArea;

  String         context = getContext ();

  ElementSet     elems   = ElementSet::get ( globdat, context );
  NodeSet        nodes   = elems.getNodes  ();

  Ref<StdShape>  shape   = newInstance<StdSquare9B> ();


  dofs_ = XDofSpace::get ( nodes.getData(), globdat );

  dofTypes_.resize ( 1 );

  dofTypes_[0] = dofs_->addType ( "u" );

  dofs_->addDofs (
    IdxVector ( iarray( nodes.size() ) ),
    dofTypes_
  );

  element_ = newInstance<BezierElement> (
    elems,
    Database::get ( "C", elems.getData(), globdat, context ),
    newInstance<ParametricArea> (
      "shape",
      StdSquare::getGaussScheme   ( 3, 3 ),
      newInstance<StdBezierShape> ( shape )
    )
  );
}


PoissonModel::~PoissonModel ()
{}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool PoissonModel::takeAction

  ( const String&      action,
    const Properties&  params,
    const Properties&  globdat )

{
  using jive::model::Actions;
  using jive::model::ActionParams;
  using jive::model::StateVector;


  if ( action == Actions::GET_TABLE )
  {
    return getTable_ ( params, globdat );
  }

  if ( action == Actions::GET_MATRIX0 )
  {
    Ref<MBuilder>  mbuilder;

    Vector         state;
    Vector         fint;

    StateVector::get ( state,    dofs_, globdat );
    params      .get ( fint,     ActionParams::INT_VECTOR );
    params      .get ( mbuilder, ActionParams::MATRIX0 );

    calcMatrix_ ( *mbuilder, fint, state );

    return true;
  }

  return false;
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Model> PoissonModel::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  return newInstance<PoissonModel> ( name, conf, props, globdat );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void PoissonModel::declare ()
{
  using jive::model::ModelFactory;

  ModelFactory::declare ( PoissonModel::TYPE_NAME,
                          & PoissonModel::makeNew );
}


//-----------------------------------------------------------------------
//   calcMatrix_
//-----------------------------------------------------------------------


void PoissonModel::calcMatrix_

  ( MBuilder&      mb,
    const Vector&  fint,
    const Vector&  state ) const

{
  using jem::numeric::matmul;

  BezierElement&  elem      = *element_;

  const idx_t     elemCount = elem.elems.size ();

  Matrix          elMat     ( elem.nodeCount, elem.nodeCount );
  Vector          elVec     ( elem.nodeCount );
  Vector          elState   ( elem.nodeCount );
  IdxVector       idofs     ( elem.nodeCount );


  elem.selectPoints ( BezierElement::IPOINTS );

  for ( idx_t ielem = 0; ielem < elemCount; ielem++ )
  {
    elem.loadData        ( ielem );
    elem.getShapeGrads   ();
    dofs_->getDofIndices ( idofs, elem.inodes, dofTypes_ );

    elMat = 0.0;

    for ( idx_t ip = 0; ip < elem.pntCount; ip++ )
    {
      Matrix  grads = elem.shapeGrads[ip];

      elMat += elem.pntWeights[ip] *
               matmul ( grads.transpose(), grads );
    }

    elState = state[idofs];
    elVec   = matmul ( elMat, elState );

    mb.addBlock ( idofs, idofs, elMat );

    fint[idofs] += elVec;
  }
}


//-----------------------------------------------------------------------
//   getTable_
//-----------------------------------------------------------------------


bool PoissonModel::getTable_

  ( const Properties&  params,
    const Properties&  globdat ) const

{
  using jive::model::ActionParams;
  using jive::model::StateVector;

  Ref<XTable>  table;
  Vector       state;
  String       name;


  params.get ( name, ActionParams::TABLE_NAME );

  if ( name == "flux" )
  {
    params.get ( table, ActionParams::TABLE );

    if ( table->getRowItems() == element_->elems.getData() )
    {
      StateVector::get ( state, dofs_, globdat );
      calcFlux_        ( *table, state );

      return true;
    }
  }

  return false;
}


//-----------------------------------------------------------------------
//   calcFlux_
//-----------------------------------------------------------------------


void PoissonModel::calcFlux_

  ( XTable&        table,
    const Vector&  state ) const

{
  using jem::numeric::matmul;

  BezierElement&  elem      = *element_;

  const idx_t     elemCount = elem.elems.size ();

  Vector          elState   ( elem.nodeCount );
  IdxVector       idofs     ( elem.nodeCount );

  IdxMatrix       jcols;
  Matrix          flux;


  elem.selectPoints ( BezierElement::VERTICES );

  jcols.resize ( elem.rank, elem.pntCount );
  flux .resize ( elem.rank, elem.pntCount );

  for ( idx_t j = 0; j < elem.pntCount; j++ )
  {
    for ( idx_t i = 0; i < elem.rank; i++ )
    {
      jcols(i,j) = table.addColumn (
        String::format ( "f(%d,%d)", i, j )
      );
    }
  }

  for ( idx_t ielem = 0; ielem < elemCount; ielem++ )
  {
    elem.loadData        ( ielem );
    elem.getShapeGrads   ();
    dofs_->getDofIndices ( idofs, elem.inodes, dofTypes_ );

    elState = state[idofs];

    for ( idx_t i = 0; i < elem.pntCount; i++ )
    {
      flux[i] = matmul ( elem.shapeGrads[i], elState );
    }

    table.setRowValues ( ielem, flatten( jcols ), flatten( flux ) );
  }
}

