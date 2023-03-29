
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
#include <jem/base/Exception.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/array/operators.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/numeric/func/Function.h>
#include <jem/util/Pattern.h>
#include <jem/util/Properties.h>
#include <jive/util/utilities.h>
#include <jive/util/Globdat.h>
#include <jive/util/FuncUtils.h>
#include <jive/model/Names.h>
#include <jive/model/Actions.h>
#include <jive/model/ModelFactory.h>
#include <jive/model/DummyModel.h>
#include <jive/model/LoadScaleModel.h>


JEM_DEFINE_SERIABLE_CLASS( jive::model::LoadScaleModel );


JIVE_BEGIN_PACKAGE( model )


using jem::newInstance;
using jem::util::Pattern;
using jive::util::FuncUtils;


//=======================================================================
//   class LoadScaleModel
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  LoadScaleModel::TYPE_NAME = "LoadScale";


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


LoadScaleModel::LoadScaleModel

  ( const String&      name,
    const String&      loadCase,
    const Ref<Model>&  child ) :

    Super     (     name ),
    child_    (    child ),
    loadCase_ ( loadCase )

{
  if ( ! child_ )
  {
    child_ = newInstance<DummyModel> ( name );
  }

  if ( myName_.size() == 0 )
  {
    myName_ = child_->getName ();
  }

  loadScale_ = 1.0;
}


LoadScaleModel::~LoadScaleModel ()
{}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void LoadScaleModel::readFrom ( ObjectInput& in )
{
  decode ( in,
           myName_,
           child_,
           loadCase_,
           loadScale_,
           scaleFunc_ );
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void LoadScaleModel::writeTo ( ObjectOutput& out ) const
{
  encode ( out,
           myName_,
           child_,
           loadCase_,
           loadScale_,
           scaleFunc_ );
}


//-----------------------------------------------------------------------
//   findModel
//-----------------------------------------------------------------------


Model* LoadScaleModel::findModel ( const String& name ) const
{
  if ( name == myName_ )
  {
    return const_cast<Self*> ( this );
  }
  else
  {
    return child_->findModel ( name );
  }
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void LoadScaleModel::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  if ( props.contains( myName_ ) )
  {
    Properties  myProps = props.findProps ( myName_ );

    myProps.find ( loadCase_,  PropNames::LOAD_CASE  );
    myProps.find ( loadScale_, PropNames::LOAD_SCALE );

    FuncUtils::configFunc ( scaleFunc_, PropNames::SCALE_FUNC,
                            myProps, globdat );
  }

  child_->configure ( props, globdat );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void LoadScaleModel::getConfig

  ( const Properties&  conf,
    const Properties&  globdat ) const

{
  Properties  myConf = conf.makeProps ( myName_ );

  myConf.set ( PropNames::LOAD_CASE,  loadCase_  );
  myConf.set ( PropNames::LOAD_SCALE, loadScale_ );

  FuncUtils::getConfig ( myConf, scaleFunc_,
                         PropNames::SCALE_FUNC );
  child_   ->getConfig ( conf, globdat );
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool LoadScaleModel::takeAction

  ( const String&      action,
    const Properties&  params,
    const Properties&  globdat )

{
  using jem::Exception;
  using jive::util::Globdat;

  if ( action == Actions::GET_EXT_VECTOR ||
       action == Actions::GET_CONSTRAINTS )
  {
    double  scale0, scale;

    scale = scale0 = 1.0;

    params.find ( scale0, ActionParams::SCALE_FACTOR );

    if ( ! scaleFunc_ )
    {
      scale = loadScale_;
    }
    else
    {
      try
      {
        scale = FuncUtils::evalFunc ( *scaleFunc_, globdat );
      }
      catch ( Exception& ex )
      {
        ex.setContext ( getContext() );
        throw;
      }
    }

    scale *= scale0;

    if ( Globdat::hasVariable( myName_, globdat ) )
    {
      Properties  vars = Globdat::getVariables ( myName_, globdat );

      vars.set ( VarNames::LOAD_SCALE, scale );
    }

    params.set ( ActionParams::SCALE_FACTOR, scale );

    child_->takeAction ( action, params, globdat );

    params.set ( ActionParams::SCALE_FACTOR, scale0 );

    return true;
  }

  if ( action == Actions::SET_LOAD_SCALE )
  {
    String  loadCase;

    if ( params.find( loadCase, ActionParams::LOAD_CASE ) )
    {
      if ( ! Pattern::matches( loadCase, loadCase_ ) )
      {
        return false;
      }
    }

    params.get ( loadScale_, ActionParams::SCALE_FACTOR );

    return true;
  }

  return child_->takeAction ( action, params, globdat );
}


//-----------------------------------------------------------------------
//   getScaleFactor
//-----------------------------------------------------------------------


double LoadScaleModel::getScaleFactor () const
{
  return loadScale_;
}


//-----------------------------------------------------------------------
//   setScaleFactor
//-----------------------------------------------------------------------


void LoadScaleModel::setScaleFactor ( double scale )
{
  loadScale_ = scale;
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Model> LoadScaleModel::makeNew

  ( const String&      myName,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  String  childName = util::joinNames ( myName, PropNames::MODEL );

  String  loadCase;

  idx_t  i = myName.rfind ( '.' );

  if ( i >= 0 )
  {
    loadCase = myName[slice(i + 1,END)];
  }
  else
  {
    loadCase = myName;
  }

  if ( loadCase.size() == 0 )
  {
    loadCase = "load";
  }

  return newInstance<Self> (
    myName,
    loadCase,
    newModel ( childName, conf, props, globdat )
  );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void LoadScaleModel::declare ()
{
  ModelFactory::declare ( TYPE_NAME,  & makeNew );
  ModelFactory::declare ( CLASS_NAME, & makeNew );
}


JIVE_END_PACKAGE( model )
