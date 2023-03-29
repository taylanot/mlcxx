
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
#include <jem/base/Exception.h>
#include <jem/base/ClassTemplate.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/Properties.h>
#include <jive/util/Table.h>
#include <jive/util/XDofSpace.h>
#include <jive/util/FuncUtils.h>
#include <jive/geom/Shape.h>
#include <jive/geom/ShapeTable.h>
#include <jive/model/Actions.h>
#include <jive/model/Globdat.h>
#include <jive/model/Model.h>
#include <jive/model/ModelFactory.h>
#include <jive/fem/Element.h>
#include <jive/fem/ElementIterator.h>
#include <jive/femodel/Names.h>
#include <jive/femodel/misc/declare.h>
#include <jive/femodel/misc/SourceModel.h>


JEM_DEFINE_SERIABLE_CLASS( jive::femodel::SourceModel );


JIVE_BEGIN_PACKAGE( femodel )


using jem::newInstance;
using jem::Exception;
using jive::util::FuncUtils;


//=======================================================================
//   class SourceModel
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  SourceModel::TYPE_NAME = "Source";


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


SourceModel::SourceModel ()
{}


SourceModel::SourceModel

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

  ElemIter&   et    = * elemIter_;
  ElementSet  elems = et.getElements ();
  NodeSet     nodes = elems.getNodes ();

  et.setShapeFlags ( ShapeTable::REGULAR_TYPE );
  et.checkShapes   ();

  nodes.checkRank  ( context, 1, 3 );

  dofName_ = "u";

  myProps.find ( dofName_, PropNames::DOF_NAME );
  myConf. set  ( PropNames::DOF_NAME, dofName_ );

  dofs_    = XDofSpace::get ( nodes.getData(), globdat );
  dofType_ = dofs_->addType ( dofName_ );

  dofs_->addDofs ( et.getNodeIndices(), dofType_ );

  tableCol_ = -1;
  element_  =  newInstance<Element> ( elems );
}


SourceModel::~SourceModel ()
{}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void SourceModel::readFrom ( ObjectInput& in )
{
  decode ( in,
           myName_,
           elemIter_,
           dofs_,
           dofName_,
           dofType_ );

  decode ( in,
           sourceFunc_,
           sourceTable_,
           tableName_,
           tableCol_ );

  element_ = newInstance<Element> ( elemIter_->getElements() );
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void SourceModel::writeTo ( ObjectOutput& out ) const
{
  encode ( out,
           myName_,
           elemIter_,
           dofs_,
           dofName_,
           dofType_ );

  encode ( out,
           sourceFunc_,
           sourceTable_,
           tableName_,
           tableCol_ );
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void SourceModel::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  if ( ! props.contains( myName_ ) )
  {
    return;
  }

  Properties  myProps = props.findProps ( myName_ );

  String      context = getContext ();
  String      name;


  elemIter_->configure ( myProps );

  if ( myProps.find( name, PropNames::SOURCE_TABLE ) )
  {
    NodeSet     nodes = element_->nodes;

    Ref<Table>  table =

      Table::get ( name, nodes.getData(), globdat, context );

    idx_t       jcol  = table->getColumnIndex ( dofName_ );

    // The data are OK, so commit the changes

    tableName_   = name;
    sourceTable_ = table;
    tableCol_    = jcol;
  }

  if ( myProps.contains( PropNames::SOURCE_FUNC ) )
  {
    const idx_t  rank = element_->rank;
    String     args = "i, t";

    if      ( rank == 1 )
    {
      args = args + ", x";
    }
    else if ( rank == 2 )
    {
      args = args + ", x, y";
    }
    else
    {
      args = args + ", x, y, z";
    }

    FuncUtils::configFunc ( sourceFunc_, args,
                            PropNames::SOURCE_FUNC,
                            myProps, globdat );
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void SourceModel::getConfig

  ( const Properties&  conf,
    const Properties&  globdat ) const

{
  Properties  myConf = conf.makeProps ( myName_ );

  myConf.set ( PropNames::SOURCE_TABLE, tableName_ );

  elemIter_->getConfig ( myConf );
  FuncUtils::getConfig ( myConf, sourceFunc_,
                         PropNames::SOURCE_FUNC );
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool SourceModel::takeAction

  ( const String&      action,
    const Properties&  params,
    const Properties&  globdat )

{
  using jive::model::Actions;
  using jive::model::ActionParams;


  if ( sourceTable_ == nullptr && sourceFunc_ == nullptr )
  {
    return false;
  }

  if ( action == Actions::GET_EXT_VECTOR )
  {
    Vector  fext;
    double  scale;

    scale = 1.0;

    params.find ( scale, ActionParams::SCALE_FACTOR );
    params.get  ( fext,  ActionParams::EXT_VECTOR   );

    try
    {
      getExtVector_ ( fext, scale, globdat );
    }
    catch ( Exception& ex )
    {
      ex.setContext ( getContext() );
      throw;
    }

    return true;
  }

  return false;
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Model> SourceModel::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  return newInstance<Self> ( name, conf, props, globdat );
}


//-----------------------------------------------------------------------
//   getExtVector_
//-----------------------------------------------------------------------


void SourceModel::getExtVector_

  ( const Vector&      fext,
    double             scale,
    const Properties&  globdat ) const

{
  using jem::dot;
  using jem::TensorIndex;
  using jive::model::Globdat;

  ElemIter&    et    = * elemIter_;
  Element&     e     = * element_;

  Function*    func  = sourceFunc_ .get ();
  Table*       table = sourceTable_.get ();

  Vector       elemSource;
  Vector       elemVec;

  IdxVector    idofs;
  TensorIndex  i;

  double       args[5];
  double       s;


  if ( func )
  {
    idx_t   istep = 0;
    double  tm    = 0.0;

    globdat.find ( istep, Globdat::TIME_STEP );
    globdat.find ( tm,    Globdat::TIME );

    args[0] = (double) istep;
    args[1] = tm;

    FuncUtils::resolve ( *func, globdat );
  }

  et.rewind ();

  while ( et.toNext( e ) )
  {
    if ( e.newShape )
    {
      elemSource.resize ( e.nodeCount );
      elemVec   .resize ( e.nodeCount );
      idofs     .resize ( e.nodeCount );
    }

    dofs_->getDofIndices ( idofs, e.inodes, dofType_ );

    e.getShapeFuncs   ();
    e.getPointWeights ();

    elemVec = 0.0;

    if ( table )
    {
      table->getColValues ( elemSource, e.inodes, tableCol_ );

      for ( idx_t ip = 0; ip < e.pntCount; ip++ )
      {
        s = dot ( elemSource, e.shapeFuncs[ip] );

        elemVec[i] +=

          (s * scale * e.pntWeights[ip]) *  e.shapeFuncs(i,ip);
      }
    }

    if ( func )
    {
      e.getGlobalPoints ();

      for ( idx_t ip = 0; ip < e.pntCount; ip++ )
      {
        for ( idx_t j = 0; j < e.rank; j++ )
        {
          args[j + 2] = e.xpoints(j,ip);
        }

        s = func->getValue ( args );

        elemVec[i] +=

          (s * scale * e.pntWeights[ip]) * e.shapeFuncs(i,ip);
      }
    }

    fext[idofs] += elemVec;
  }
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   declareSourceModel
//-----------------------------------------------------------------------


void declareSourceModel ()
{
  using jive::model::ModelFactory;

  ModelFactory::declare ( SourceModel::TYPE_NAME,
                          & SourceModel::makeNew );

  ModelFactory::declare ( SourceModel::CLASS_NAME,
                          & SourceModel::makeNew );
}


JIVE_END_PACKAGE( femodel )
