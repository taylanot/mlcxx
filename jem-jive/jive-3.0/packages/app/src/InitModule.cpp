
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


#include <jem/base/System.h>
#include <jem/base/Float.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/array/utilities.h>
#include <jem/base/IllegalInputException.h>
#include <jem/util/None.h>
#include <jem/util/Properties.h>
#include <jive/util/utilities.h>
#include <jive/util/Table.h>
#include <jive/util/TableUtils.h>
#include <jive/util/TableException.h>
#include <jive/util/ItemSet.h>
#include <jive/util/XDofSpace.h>
#include <jive/util/Constraints.h>
#include <jive/util/ConstraintsParser.h>
#include <jive/util/VectorManager.h>
#include <jive/mp/utilities.h>
#include <jive/mp/Globdat.h>
#include <jive/mp/BorderSet.h>
#include <jive/mp/TableExchanger.h>
#include <jive/mp/VectorExchanger.h>
#include <jive/model/Actions.h>
#include <jive/model/MPModel.h>
#include <jive/app/import.h>
#include <jive/app/Names.h>
#include <jive/app/Globdat.h>
#include <jive/app/ModuleFactory.h>
#include <jive/app/InitModule.h>


JEM_DEFINE_CLASS( jive::app::InitModule );


JIVE_BEGIN_PACKAGE( app )


using jem::dynamicCast;
using jem::newInstance;
using jem::System;
using jem::IllegalInputException;
using jive::util::joinNames;
using jive::util::ItemSet;
using jive::util::XDofSpace;
using jive::model::Model;


//=======================================================================
//   class InitModule
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


InitModule::InitModule ( const String& name ) :

  Super ( name )

{}


InitModule::~InitModule ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


Module::Status InitModule::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  makeModel_       ( conf, props, globdat );
  reorderDofs_     ( conf, props, globdat );
  syncDofs_        ( conf, props, globdat );
  initConstraints_ ( conf, props, globdat );
  initVectors_     ( conf, props, globdat );
  initModel_       ( conf, props, globdat );

  return DONE;
}


//-----------------------------------------------------------------------
//   shutdown
//-----------------------------------------------------------------------


void InitModule::shutdown ( const Properties& globdat )
{
  using jive::model::Actions;

  // Remove the model from the global database to break possible
  // circular references.

  Ref<Model>  model = Model::extract ( globdat );

  if ( model )
  {
    Properties  params;

    model->takeAction ( Actions::SHUTDOWN, params, globdat );
  }
}

//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void InitModule::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  Ref<Model>  model = Model::find ( globdat );

  if ( model )
  {
    model->configure ( props, globdat );
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void InitModule::getConfig

  ( const Properties&  props,
    const Properties&  globdat ) const

{
  Ref<Model>  model = Model::find ( globdat );

  if ( model )
  {
    model->getConfig ( props, globdat );
  }
}


//-----------------------------------------------------------------------
//   makeModel
//-----------------------------------------------------------------------


void InitModule::makeModel

  ( const String&       name,
    const Properties&   conf,
    const Properties&   props,
    const Properties&   globdat,
    const Ref<Writer>&  logger )

{
  using jem::util::isNone;
  using jive::mp::BorderSet;
  using jive::mp::TableExchanger;
  using jive::mp::VectorExchanger;
  using jive::model::newModel;
  using jive::model::MPModel;
  using jive::model::Actions;
  using jive::model::ActionParams;

  Ref<Model>  model = Model::find ( globdat );


  if ( ! model )
  {
    model = newModel ( name, conf, props, globdat );

    if ( logger )
    {
      print ( *logger, "Created a new model named `",
              name, "\'.\n" );
    }

    // Automatically wrap the model in an MP model if required. A fake
    // do-loop is used here to avoid goto's and deeply nested
    // if-statements.

    do
    {
      // Check whether this is a parallel computation.

      if ( mp::Globdat::procCount( globdat ) <= 1 )
      {
        break;
      }

      // Try to find the primary DOF space.

      Ref<DofSpace>  dofs = DofSpace::find ( globdat );

      if ( ! dofs )
      {
        break;
      }

      Ref<ItemSet>  items = dofs->getItems ();

      // Check whether any exchange borders have been defined.

      if ( ! BorderSet::hasBorders( items, globdat ) )
      {
        break;
      }

      Properties  params ( "modelParams" );
      Ref<Model>  tmpModel;

      // Check whether an MP model has already been defined.

      model->takeAction ( Actions::GET_MP_MODEL, params, globdat );
      params.find       ( tmpModel, ActionParams::MP_MODEL );

      if ( tmpModel )
      {
        break;
      }

      // Check whether the "exchange" property has been set to none.

      String       xname = joinNames ( name, "exchange" );
      Ref<Object>  obj;

      if ( props.find( obj, xname ) && isNone( obj ) )
      {
        break;
      }

      // OK, wrap the model in an MP model.

      Ref<TableExchanger>   tex =

        TableExchanger::get ( items, globdat );

      Ref<VectorExchanger>  vex =

        VectorExchanger::get ( dofs, globdat );

      Ref<MPModel>          mpModel =

        newInstance<MPModel> ( xname, model, tex, vex );

      mpModel->setDefaults ();

      model = mpModel;

      // Show what is going on.

      if ( logger )
      {
        print ( *logger, "Wrapped model `", name,
                "\' in an MPModel named `", xname, "\'.\n\n" );
      }
    }
    while ( false );

    model->store ( globdat );
  }
}


//-----------------------------------------------------------------------
//   syncDofs
//-----------------------------------------------------------------------


void InitModule::syncDofs ( const Properties& globdat )
{
  StringVector    list = ItemSet::listAll ( globdat );

  Ref<DofSpace>   dofs;
  Ref<XDofSpace>  xdofs;
  Ref<ItemSet>    items;
  idx_t           i, n;


  // To be sure that all processes walk through the item set list
  // in the same order:

  jem::sort ( list );

  for ( i = 0, n = list.size(); i < n; i++ )
  {
    items = ItemSet::find ( list[i], globdat );

    if ( items )
    {
      dofs  = DofSpace::find ( items, globdat );
      xdofs = dynamicCast<XDofSpace> ( dofs );

      if ( xdofs )
      {
        mp::syncDofs ( *xdofs, globdat );

        xdofs->trimToSize ();
      }
    }
  }
}


//-----------------------------------------------------------------------
//   initConstraints
//-----------------------------------------------------------------------


void InitModule::initConstraints

  ( const Properties&  globdat,
    const String&      context )

{
  using jive::util::Constraints;
  using jive::util::ConstraintsParser;

  StringVector            list = ItemSet::listAll ( globdat );

  Ref<ConstraintsParser>  conParser;
  Ref<Constraints>        cons;
  Ref<DofSpace>           dofs;
  Ref<ItemSet>            items;
  idx_t                   i, n;


  for ( i = 0, n = list.size(); i < n; i++ )
  {
    items = ItemSet::find ( list[i], globdat );

    if ( ! items )
    {
      continue;
    }

    conParser = ConstraintsParser::extract ( items, globdat );

    if ( conParser )
    {
      dofs = DofSpace::find ( items, globdat );

      if ( dofs )
      {
        cons = Constraints::get   ( dofs, globdat );

        conParser->getConstraints ( *cons, context );

        cons->compress            ();
      }
    }
  }
}


//-----------------------------------------------------------------------
//   initVectors
//-----------------------------------------------------------------------


void InitModule::initVectors

  ( const StringVector&  specs,
    const Properties&    globdat,
    const String&        context,
    const Ref<Writer>&   logger )

{
  using jive::util::TableUtils;
  using jive::util::TableException;
  using jive::util::VectorManager;

  Ref<VectorManager>  vman;

  Ref<DofSpace>       dofs;
  Ref<ItemSet>        items;
  Ref<Table>          table;

  Vector              vec;

  String              itemPath;
  String              vecName;
  String              tabName;
  String              path;

  double              x;
  idx_t               i, k, n;


  n = specs.size ();

  for ( i = 0; i < n; i++ )
  {
    parseVecSpec_ ( itemPath, vecName, tabName, specs[i] );

    if ( vecName == "state0" )
    {
      print ( System::warn(),
              "Renaming vector `state0\' to `state\'\n\n" );

      vecName = "state";
    }

    if ( ! dofs || path != itemPath )
    {
      path = itemPath;

      if ( path.size() == 0 )
      {
        dofs  = DofSpace::get  ( globdat, context );
        items = dofs->getItems ();
      }
      else
      {
        items = ItemSet ::get ( path,  globdat, context );
        dofs  = DofSpace::get ( items, globdat, context );
      }

      vman = VectorManager::get ( dofs, globdat );
    }

    if ( vman->findVector( vec, vecName ) )
    {
      if ( logger )
      {
        print ( *logger, "  skipping vector `", vecName, "\'\n" );
      }

      continue;
    }

    // Important: allocate a *new* vector to avoid aliasing

    vec.ref ( Vector( dofs->dofCount() ) );

    if ( jem::Float::parse( x, tabName ) )
    {
      vec = x;
    }
    else
    {
      table = Table::get ( tabName, items, globdat, context );
      vec   = 0.0;
      k     = 0;

      try
      {
        k = TableUtils::toVector ( vec, *table, *dofs );
      }
      catch ( const TableException& ex )
      {
        String  errMsg = ex.what ();

        throw IllegalInputException (
          context,
          String::format (
            "error initializing vector `%s\': %s",
            vecName,
            errMsg
          )
        );
      }

      if ( k != vec.size() )
      {
        print ( System::warn(),
                "Table `", tabName,
                "\' does not set all elements of the "
                "vector `", vecName,
                "\';\n  non-set elements will be zero\n\n" );
      }
    }

    vman->storeVector ( vecName, vec );

    if ( logger )
    {
      print ( *logger, "  ", specs[i], '\n' );
    }
  }

  if ( n > 0 && logger )
  {
    print ( *logger, '\n' );
  }
}


//-----------------------------------------------------------------------
//   checkVecSpecs
//-----------------------------------------------------------------------


void InitModule::checkVecSpecs ( const StringVector& specs )
{
  const idx_t  n = specs.size ();

  String       itemPath;
  String       vecName;
  String       tabName;


  for ( idx_t i = 0; i < n; i++ )
  {
    parseVecSpec_ ( itemPath, vecName, tabName, specs[i] );
  }
}


//-----------------------------------------------------------------------
//   initModel
//-----------------------------------------------------------------------


void InitModule::initModel

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  using jive::model::Actions;


  Ref<Model>  model = Model::find ( globdat );

  if ( model )
  {
    model->configure  ( props, globdat );
    model->getConfig  ( conf,  globdat );
    model->takeAction ( Actions::INIT, Properties(), globdat );
  }
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Module> InitModule::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  return newInstance<Self> ( name );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void InitModule::declare ()
{
  ModuleFactory::declare ( "Init",     & makeNew );
  ModuleFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   makeModel_
//-----------------------------------------------------------------------


void InitModule::makeModel_

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  String  name;
  idx_t   i;


  i = myName_.rfind ( '.' );

  if ( i >= 0 )
  {
    name = joinNames ( myName_[slice(BEGIN,i)], PropNames::MODEL );
  }
  else
  {
    name = PropNames::MODEL;
  }

  makeModel ( name, conf, props, globdat,
              & System::info( myName_ ) );
}


//-----------------------------------------------------------------------
//   reorderDofs_
//-----------------------------------------------------------------------


void InitModule::reorderDofs_

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{}


//-----------------------------------------------------------------------
//   syncDofs_
//-----------------------------------------------------------------------


void InitModule::syncDofs_

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  syncDofs ( globdat );
}


//-----------------------------------------------------------------------
//   initConstraints_
//-----------------------------------------------------------------------


void InitModule::initConstraints_

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  initConstraints ( globdat, getContext() );
}


//-----------------------------------------------------------------------
//   initVectors_
//-----------------------------------------------------------------------


void InitModule::initVectors_

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  Properties    myConf  = conf .makeProps ( myName_ );
  Properties    myProps = props.findProps ( myName_ );

  StringVector  vecSpecs;


  myProps.find ( vecSpecs, PropNames::VECTORS );

  if ( vecSpecs.size() > 0 )
  {
    try
    {
      checkVecSpecs ( vecSpecs );
    }
    catch ( IllegalInputException& ex )
    {
      ex.setContext ( myProps.getContext( PropNames::VECTORS ) );
      throw;
    }

    print ( System::out(), "Initializing (state) vectors ...\n\n" );

    initVectors ( vecSpecs,
                  globdat,
                  getContext     (),
                  & System::info ( myName_ ) );
  }

  myConf.set ( PropNames::VECTORS, vecSpecs );
}


//-----------------------------------------------------------------------
//   initModel_
//-----------------------------------------------------------------------


void InitModule::initModel_

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  initModel ( conf, props, globdat );
}


//-----------------------------------------------------------------------
//   parseVecSpec_
//-----------------------------------------------------------------------


void InitModule::parseVecSpec_

  ( String&        itemPath,
    String&        vecName,
    String&        tabName,
    const String&  vecSpec )

{
  idx_t  i = vecSpec.rfind ( '=' );

  if ( i < 0 )
  {
    vecName = vecSpec;
  }
  else
  {
    vecName = vecSpec[slice(BEGIN,i)];
    tabName = vecSpec[slice(i + 1,END)].stripWhite ();

    if ( tabName.size() == 0 )
    {
      throw IllegalInputException (
        JEM_FUNC,
        "parse error after `=\': missing table name or number"
      );
    }
  }

  i = vecName.find ( '/' );

  if ( i < 0 )
  {
    vecName  = vecName.stripWhite ();
  }
  else
  {
    itemPath = vecName[slice(BEGIN,i)]  .stripWhite ();
    vecName  = vecName[slice(i + 1,END)].stripWhite ();
  }

  if ( vecName.size() == 0 )
  {
    throw IllegalInputException ( JEM_FUNC, "empty vector name" );
  }
}


JIVE_END_PACKAGE( app )
