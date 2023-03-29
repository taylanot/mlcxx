
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
#include <jive/femodel/misc/LoadModel.h>


JEM_DEFINE_SERIABLE_CLASS( jive::femodel::LoadModel );


JIVE_BEGIN_PACKAGE( femodel )


using jem::newInstance;
using jem::Array;
using jive::util::FuncUtils;


//=======================================================================
//   class LoadModel
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  LoadModel::TYPE_NAME = "Load";


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


LoadModel::LoadModel ()
{}


LoadModel::LoadModel

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

  myProps.get ( dofNames_, PropNames::DOFS );
  myConf. set ( PropNames::DOFS, dofNames_ );

  dofs_ = XDofSpace::get ( nodes.getData(), globdat );

  dofTypes_.resize ( dofNames_.size() );

  for ( idx_t i = 0; i < dofNames_.size(); i++ )
  {
    dofTypes_[i] = dofs_->addType ( dofNames_[i] );
  }

  dofs_->addDofs ( et.getNodeIndices(), dofTypes_ );

  element_ = newInstance<Element> ( elems );
}


LoadModel::~LoadModel ()
{}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void LoadModel::readFrom ( ObjectInput& in )
{
  decode ( in,
           myName_,
           elemIter_,
           dofs_,
           dofNames_,
           dofTypes_,
           loads_ );

  element_ = newInstance<Element> ( elemIter_->getElements() );
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void LoadModel::writeTo ( ObjectOutput& out ) const
{
  encode ( out,
           myName_,
           elemIter_,
           dofs_,
           dofNames_,
           dofTypes_,
           loads_ );
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void LoadModel::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  using jem::Array;

  if ( ! props.contains( myName_ ) )
  {
    return;
  }

  Properties  myProps = props.findProps ( myName_ );


  elemIter_->configure ( myProps );

  if ( myProps.contains( PropNames::LOAD ) )
  {
    Array< Ref<Function> >  loads;

    const idx_t  rank = element_->rank;
    String       args = "i, t";

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

    FuncUtils::configFuncs ( loads, args,
                             PropNames::LOAD,
                             myProps, globdat );

    if ( loads.size() > 0 && loads.size() != dofNames_.size() )
    {
      myProps.propertyError (
        PropNames::LOAD,
        String::format (
          "load vector has size %d but should have "
          "size zero or size %d",
          loads    .size (),
          dofNames_.size ()
        )
      );
    }

    loads_.swap ( loads );
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void LoadModel::getConfig

  ( const Properties&  conf,
    const Properties&  globdat ) const

{
  Properties  myConf = conf.makeProps ( myName_ );

  elemIter_->getConfig ( myConf );
  FuncUtils::getConfig ( myConf, loads_, PropNames::LOAD );
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool LoadModel::takeAction

  ( const String&      action,
    const Properties&  params,
    const Properties&  globdat )

{
  using jem::Exception;
  using jive::model::Actions;
  using jive::model::ActionParams;


  if ( loads_.size() == 0 )
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
      FuncUtils::resolve ( loads_,      globdat );
      getExtVector_      ( fext, scale, globdat );
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


Ref<Model> LoadModel::makeNew

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


void LoadModel::getExtVector_

  ( const Vector&      fext,
    double             scale,
    const Properties&  globdat ) const

{
  using jem::dot;
  using jem::TensorIndex;
  using jive::model::Globdat;

  const idx_t  n  = dofTypes_.size ();

  ElemIter&    et = * elemIter_;
  Element&     e  = * element_;

  Vector       fe;
  IdxVector    idofs;

  TensorIndex  i;

  double       args[5];
  double       s, t, w;

  idx_t        j;


  j = 0;
  t = 0.0;

  globdat.find ( j, Globdat::TIME_STEP );
  globdat.find ( t, Globdat::TIME );

  args[0] = (double) j;
  args[1] = t;

  et.rewind ();

  while ( et.toNext( e ) )
  {
    if ( e.newShape )
    {
      fe   .resize ( n * e.nodeCount );
      idofs.resize ( n * e.nodeCount );
    }

    dofs_->getDofIndices ( idofs, e.inodes, dofTypes_ );

    e.getShapeFuncs   ();
    e.getPointWeights ();
    e.getGlobalPoints ();

    fe = 0.0;

    for ( idx_t ip = 0; ip < e.pntCount; ip++ )
    {
      for ( j = 0; j < e.rank; j++ )
      {
        args[j + 2] = e.xpoints(j,ip);
      }

      w = scale * e.pntWeights[ip];

      for ( j = 0; j < n; j++ )
      {
        s = w * loads_[j]->getValue ( args );

        fe[i * n + j] += s * e.shapeFuncs(i,ip);
      }
    }

    fext[idofs] += fe;
  }
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   declareLoadModel
//-----------------------------------------------------------------------


void declareLoadModel ()
{
  using jive::model::ModelFactory;

  ModelFactory::declare ( LoadModel::TYPE_NAME,
                          & LoadModel::makeNew );

  ModelFactory::declare ( LoadModel::CLASS_NAME,
                          & LoadModel::makeNew );
}


JIVE_END_PACKAGE( femodel )
