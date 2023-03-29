
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
#include <jem/base/array/tensor.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalInputException.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/numeric/algebra/matmul.h>
#include <jem/util/Properties.h>
#include <jive/util/ItemGroup.h>
#include <jive/util/XTable.h>
#include <jive/util/XDofSpace.h>
#include <jive/util/FuncUtils.h>
#include <jive/algebra/MatrixBuilder.h>
#include <jive/geom/ShapeTable.h>
#include <jive/model/ModelFactory.h>
#include <jive/model/Actions.h>
#include <jive/model/StateVector.h>
#include <jive/fem/InternalElement.h>
#include <jive/fem/ElementIterator.h>
#include <jive/femodel/Names.h>
#include <jive/femodel/misc/declare.h>
#include <jive/femodel/misc/TransportModel.h>


JEM_DEFINE_SERIABLE_CLASS( jive::femodel::TransportModel );


JIVE_BEGIN_PACKAGE( femodel )


using jem::dot;
using jem::select;
using jem::newInstance;
using jem::TensorIndex;
using jem::numeric::matmul;


//=======================================================================
//   class TransportModel
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  TransportModel::TYPE_NAME      = "Transport";
const char*  TransportModel::COORD_NAMES[3] = { "x", "y", "z" };


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


TransportModel::TransportModel ()
{}


TransportModel::TransportModel

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat ) :

    Super ( name )

{
  using jive::geom::ShapeTable;

  Properties    myConf  = conf .makeProps ( myName_ );
  Properties    myProps = props.findProps ( myName_ );

  const String  context = getContext ();


  elemIter_ = newInstance<ElemIter> ( myConf,  myProps,
                                      globdat, context );

  ElemIter&     et    = * elemIter_;
  ElementSet    elems = et.getElements ();
  NodeSet       nodes = elems.getNodes ();

  const idx_t   rank  = nodes.rank     ();

  et.setShapeFlags ( ShapeTable::INTERNAL_TYPE |
                     ShapeTable::REGULAR_TYPE );
  et.checkShapes   ();

  nodes.checkRank  ( context, 1, 3 );

  dofName_ = "u";

  myProps.find ( dofName_, PropNames::DOF_NAME );
  myConf. set  ( PropNames::DOF_NAME, dofName_ );

  dofs_    = XDofSpace::get ( nodes.getData(), globdat );
  dofType_ = dofs_->addType ( dofName_ );

  dofs_->addDofs ( et.getNodeIndices(), dofType_ );

  flowCols_ .resize ( rank );
  diffusion_.resize ( rank );

  flowCols_  = -1_idx;
  diffusion_ =  1.0;

  element_   = newInstance<IElement> ( elems );
}


TransportModel::~TransportModel ()
{}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void TransportModel::readFrom ( ObjectInput& in )
{
  decode ( in,
           myName_,
           elemIter_,
           dofs_,
           dofName_,
           dofType_ );

  decode ( in,
           flowTable_,
           flowName_,
           flowCols_,
           diffusion_ );

  element_ = newInstance<IElement> ( elemIter_->getElements() );
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void TransportModel::writeTo ( ObjectOutput& out ) const
{
  encode ( out,
           myName_,
           elemIter_,
           dofs_,
           dofName_,
           dofType_ );

  encode ( out,
           flowTable_,
           flowName_,
           flowCols_,
           diffusion_ );
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void TransportModel::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  if ( ! props.contains( myName_ ) )
  {
    return;
  }

  NodeSet      nodes   = element_->nodes;

  const idx_t  rank    = element_->rank;

  Properties   myProps = props.findProps ( myName_ );

  Vector       diff;
  String       name;


  elemIter_->configure ( myProps );

  if ( myProps.find( name, PropNames::FLOW_TABLE ) )
  {
    Ref<Table>  field =

      Table::get ( name, nodes.getData(), globdat, getContext() );

    IdxVector   jcols ( rank );

    for ( idx_t r = 0; r < rank; r++ )
    {
      jcols[r] = field->getColumnIndex ( COORD_NAMES[r] );
    }

    // The data are OK, so commit the changes

    flowName_  = name;
    flowTable_ = field;

    flowCols_.swap ( jcols );
  }

  if ( myProps.find( diff, PropNames::DIFFUSION ) )
  {
    if      ( diff.size() == 1 )
    {
      diffusion_ = diff[0];
    }
    else if ( diff.size() == rank )
    {
      diffusion_ = diff;
    }
    else
    {
      myProps.propertyError (
        PropNames::DIFFUSION,
        String::format (
          "expected one or %d diffusion coefficients", rank
        )
      );
    }
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void TransportModel::getConfig

  ( const Properties&  conf,
    const Properties&  globdat ) const

{
  Properties  myConf = conf.makeProps ( myName_ );

  elemIter_->getConfig ( myConf );

  myConf.set ( PropNames::FLOW_TABLE, flowName_ );
  myConf.set ( PropNames::DIFFUSION,  diffusion_ );
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool TransportModel::takeAction

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

  if ( action == Actions::GET_INT_VECTOR )
  {
    Vector  state;
    Vector  fint;

    StateVector::get ( state, dofs_, globdat );
    params      .get ( fint, ActionParams::INT_VECTOR );

    getIntVector_ ( fint, state );

    return true;
  }

  if ( action == Actions::GET_MATRIX0 )
  {
    Ref<MBuilder>  mbld;

    Vector         state;
    Vector         fint;

    StateVector::get ( state, dofs_, globdat );
    params      .get ( fint,  ActionParams::INT_VECTOR );
    params      .get ( mbld,  ActionParams::MATRIX0 );

    getMatrix0_ ( *mbld, fint, state );

    return true;
  }

  if ( action == Actions::GET_MATRIX1 )
  {
    Ref<MBuilder>  mbld;

    params.get  (  mbld, ActionParams::MATRIX1 );
    getMatrix1_ ( *mbld );

    return true;
  }

  return false;
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Model> TransportModel::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  return newInstance<Self> ( name, conf, props, globdat );
}


//-----------------------------------------------------------------------
//   getIntVector_
//-----------------------------------------------------------------------


void TransportModel::getIntVector_

  ( const Vector&  fint,
    const Vector&  state ) const

{
  Table*       flow = flowTable_.get ();

  ElemIter&    et   = * elemIter_;
  IElement&    e    = * element_;

  Matrix       elemFlow;
  Matrix       ipFlow;

  Vector       stateGrads;
  Vector       elemState;
  Vector       elemVec;

  IdxVector    idofs;

  TensorIndex  i, k;


  e .selectPoints ( Element::IPOINTS );
  et.rewind       ();

  while ( et.toNext( e ) )
  {
    if ( e.newShape )
    {
      elemFlow  .resize ( e.nodeCount, e.rank     );
      ipFlow    .resize ( e.rank,      e.pntCount );
      stateGrads.resize ( e.rank      );
      elemState .resize ( e.nodeCount );
      elemVec   .resize ( e.nodeCount );
      idofs     .resize ( e.nodeCount );

      ipFlow = 0.0;
    }

    dofs_->getDofIndices ( idofs, e.inodes, dofType_ );
    e.getShapeGrads      ();

    if ( flow )
    {
      flow->getBlock ( elemFlow, e.inodes, flowCols_ );

      matmul ( ipFlow, elemFlow.transpose(), e.shapeFuncs );
    }

    elemState = select ( state, idofs );
    elemVec   = 0.0;

    for ( idx_t ip = 0; ip < e.pntCount; ip++ )
    {
      Matrix  shapeGrads = e.shapeGrads[ip];

      matmul ( stateGrads, shapeGrads, elemState );

      elemVec[i] += e.pntWeights[ip] *

        dot ( diffusion_[k] * shapeGrads(k,i), stateGrads[k], k );

      if ( flow )
      {
        const double  fdiv = dot ( ipFlow[ip], stateGrads );

        elemVec[i] += e.pntWeights[ip] * fdiv * e.shapeFuncs(i,ip);
      }
    }

    select ( fint, idofs ) += elemVec;
  }
}


//-----------------------------------------------------------------------
//   getMatrix0_
//-----------------------------------------------------------------------


void TransportModel::getMatrix0_

  ( MBuilder&      mbld,
    const Vector&  fint,
    const Vector&  state ) const

{
  Table*       flow = flowTable_.get ();

  ElemIter&    et   = * elemIter_;
  IElement&    e    = * element_;

  Matrix       elemFlow;
  Matrix       elemMat;
  Matrix       ipFlow;

  Vector       elemState;
  Vector       elemVec;
  IdxVector    idofs;

  TensorIndex  i, j, k, ip;


  e .selectPoints ( Element::IPOINTS );
  et.rewind       ();

  while ( et.toNext( e ) )
  {
    if ( e.newShape )
    {
      elemFlow .resize ( e.nodeCount, e.rank      );
      elemMat  .resize ( e.nodeCount, e.nodeCount );
      ipFlow   .resize ( e.rank,      e.pntCount  );
      elemState.resize ( e.nodeCount );
      elemVec  .resize ( e.nodeCount );
      idofs    .resize ( e.nodeCount );

      ipFlow = 0.0;
    }

    dofs_->getDofIndices ( idofs, e.inodes, dofType_ );
    e.getShapeGrads      ();

    elemMat(i,j) =

      dot ( e.pntWeights[ip],
            dot ( diffusion_[k] * e.shapeGrads(k,i,ip),
                  e.shapeGrads(k,j,ip), k ), ip );

    if ( flow )
    {
      flow->getBlock ( elemFlow, e.inodes, flowCols_ );

      matmul ( ipFlow, elemFlow.transpose(), e.shapeFuncs );

      elemMat(i,j) +=

        dot ( e.pntWeights[ip],
              e.shapeFuncs(i,ip) *
              dot ( ipFlow(k,ip), e.shapeGrads(k,j,ip), k ), ip );
    }

    mbld.addBlock ( idofs, idofs, elemMat );

    elemState    = state[idofs];

    matmul ( elemVec, elemMat, elemState );

    fint[idofs] += elemVec;
  }
}


//-----------------------------------------------------------------------
//   getMatrix1_
//-----------------------------------------------------------------------


void TransportModel::getMatrix1_ ( MBuilder&  mbld ) const
{
  ElemIter&    et = * elemIter_;
  Element&     e  = * element_;

  Matrix       elemMat;
  IdxVector    idofs;
  TensorIndex  i, j, ip;


  e .selectPoints ( Element::IPOINTS );
  et.rewind       ();

  while ( et.toNext( e ) )
  {
    if ( e.newShape )
    {
      elemMat.resize ( e.nodeCount, e.nodeCount );
      idofs  .resize ( e.nodeCount );
    }

    dofs_->getDofIndices ( idofs, e.inodes, dofType_ );

    e.getShapeFuncs   ();
    e.getPointWeights ();

    elemMat(i,j) =

      dot ( e.pntWeights[ip],
            e.shapeFuncs(i,ip) * e.shapeFuncs(j,ip), ip );

    mbld.addBlock ( idofs, idofs, elemMat );
  }
}


//-----------------------------------------------------------------------
//   getTable_
//-----------------------------------------------------------------------


bool TransportModel::getTable_

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
    NodeSet  nodes = elemIter_->getNodes ();

    params.get ( table, ActionParams::TABLE );

    if ( table->getRowItems() == nodes.getData() )
    {
      Ref<ItemGroup>  irows;
      Vector          weights;

      params.find      ( irows,   ActionParams::TABLE_ROWS    );
      params.get       ( weights, ActionParams::TABLE_WEIGHTS );

      StateVector::get ( state, dofs_, globdat );

      getNodeFluxes_   ( *table, weights, state, irows.get() );

      return true;
    }
  }

  return false;
}


//-----------------------------------------------------------------------
//   getNodeFluxes_
//-----------------------------------------------------------------------


void TransportModel::getNodeFluxes_

  ( XTable&           table,
    const Vector&     weights,
    const Vector&     globState,
    const ItemGroup*  irows ) const

{
  ElemIter&    et   = * elemIter_;
  IElement&    e    = * element_;

  Table*       flow = flowTable_.get ();

  Matrix       elemFlow;
  Matrix       elemFlux;

  Vector       stateGrads;
  Vector       elemState;

  IdxVector    jcols ( e.rank );
  IdxVector    idofs;

  TensorIndex  k;
  idx_t        i, j;


  for ( j = 0; j < e.rank; j++ )
  {
    jcols[j] = table.addColumn ( COORD_NAMES[j] );
  }

  e .selectPoints ( Element::VERTICES );
  et.rewind       ();

  while ( et.toNext( e ) )
  {
    if ( e.newShape )
    {
      elemFlow  .resize ( e.nodeCount, e.rank );
      elemFlux  .resize ( e.nodeCount, e.rank );
      stateGrads.resize ( e.rank      );
      elemState .resize ( e.nodeCount );
      idofs     .resize ( e.nodeCount );

      elemFlow = 0.0;
    }

    dofs_->getDofIndices ( idofs, e.inodes, dofType_ );
    e.getShapeGrads      ();

    if ( flow )
    {
      flow->getBlock ( elemFlow, e.inodes, flowCols_ );
    }

    if ( irows )
    {
      irows->filter ( e.inodes );
    }

    elemState = globState[idofs];
    j         = 0;

    for ( i = 0; i < e.nodeCount; i++ )
    {
      if ( e.inodes[i] < 0 )
      {
        continue;
      }

      matmul ( stateGrads, e.shapeGrads[i], elemState );

      elemFlux(j,k) = -diffusion_[k] * stateGrads[k] +
                       elemFlow(i,k) * elemState[i];

      e.inodes[j++] = e.inodes[i];
    }

    weights[e.inodes[slice(BEGIN,j)]] += 1.0;

    table.addBlock ( e.inodes[slice(BEGIN,j)], jcols,
                     elemFlux(slice(BEGIN,j),ALL) );
  }
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   declareTransportModel
//-----------------------------------------------------------------------


void declareTransportModel ()
{
  using jive::model::ModelFactory;

  ModelFactory::declare ( TransportModel::TYPE_NAME,
                          & TransportModel::makeNew );

  ModelFactory::declare ( TransportModel::CLASS_NAME,
                          & TransportModel::makeNew );
}


JIVE_END_PACKAGE( femodel )
