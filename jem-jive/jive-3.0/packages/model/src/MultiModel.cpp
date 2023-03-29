
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
#include <jem/base/Thread.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/CancelledException.h>
#include <jem/base/IllegalArgumentException.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/Properties.h>
#include <jive/util/utilities.h>
#include <jive/model/Names.h>
#include <jive/model/ModelFactory.h>
#include <jive/model/MultiModel.h>


JEM_DEFINE_SERIABLE_CLASS( jive::model::MultiModel );


JIVE_BEGIN_PACKAGE( model )


using jem::newInstance;


//=======================================================================
//   class MultiModel
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  MultiModel::TYPE_NAME = "Multi";


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


MultiModel::MultiModel ( const String& name ) :

  Super ( name )

{}


MultiModel::~MultiModel ()
{}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void MultiModel::readFrom ( ObjectInput& in )
{
  decode ( in, myName_, models_ );
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void MultiModel::writeTo ( ObjectOutput& out ) const
{
  encode ( out, myName_, models_ );
}


//-----------------------------------------------------------------------
//   findModel
//-----------------------------------------------------------------------


Model* MultiModel::findModel ( const String& name ) const
{
  if ( name == myName_ )
  {
    return const_cast<Self*> ( this );
  }
  else
  {
    const idx_t  n   = models_.size ();

    Model*       mod = 0;

    for ( idx_t i = 0; i < n; i++ )
    {
      mod = models_.getAs<Model>(i)->findModel ( name );

      if ( mod )
      {
        break;
      }
    }

    return mod;
  }
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void MultiModel::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  for ( idx_t i = 0; i < models_.size(); i++ )
  {
    Ref<Model>  mod = models_.getAs<Model> ( i );

    mod->configure ( props, globdat );
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void MultiModel::getConfig

  ( const Properties&  props,
    const Properties&  globdat ) const

{
  for ( idx_t i = 0; i < models_.size(); i++ )
  {
    Ref<Model>  mod = models_.getAs<Model> ( i );

    mod->getConfig ( props, globdat );
  }
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool MultiModel::takeAction

  ( const String&      action,
    const Properties&  params,
    const Properties&  globdat )

{
  bool  result = false;


  if ( jem::Thread::cancelled() )
  {
    throw jem::CancelledException (
      getContext (),
      String::format ( "action `%s\' cancelled", action )
    );
  }

  for ( idx_t i = 0; i < models_.size(); i++ )
  {
    Ref<Model>  mod = models_.getAs<Model> ( i );

    if ( mod->takeAction( action, params, globdat ) )
    {
      result = true;
    }
  }

  return result;
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void MultiModel::clear ()
{
  models_.clear ();
}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


void MultiModel::reserve ( idx_t n )
{
  models_.reserve ( n );
}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


void MultiModel::trimToSize ()
{
  models_.trimToSize ();
}


//-----------------------------------------------------------------------
//   addModel
//-----------------------------------------------------------------------


void MultiModel::addModel ( const Ref<Model>&  model )
{
  JEM_PRECHECK ( model );

  models_.pushBack ( model );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Model> MultiModel::makeNew

  ( const String&      myName,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  using jive::util::joinNames;


  Properties       myConf  = conf .makeProps ( myName );
  Properties       myProps = props.findProps ( myName );

  Ref<MultiModel>  multi;
  StringVector     models;
  String           modName;
  Ref<Model>       mod;
  idx_t            n;


  myProps.get ( models, PropNames::MODELS );
  myConf .set ( PropNames::MODELS, models );

  multi = newInstance<MultiModel> ( myName );
  n     = models.size ();

  for ( idx_t i = 0; i < n; i++ )
  {
    modName = models[i];

    if ( modName.size() == 0 )
    {
      myProps.propertyError ( PropNames::MODELS, "empty model name" );
    }

    mod = explicitNewModel ( joinNames ( myName, modName ),
                             conf,
                             props,
                             globdat );

    multi->addModel ( mod );
  }

  multi->trimToSize ();

  return multi;
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void MultiModel::declare ()
{
  ModelFactory::declare ( TYPE_NAME,  & makeNew );
  ModelFactory::declare ( CLASS_NAME, & makeNew );
}


JIVE_END_PACKAGE( model )
