
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
#include <jem/base/System.h>
#include <jem/base/ClassTemplate.h>
#include <jem/io/PrintWriter.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/Properties.h>
#include <jive/util/utilities.h>
#include <jive/util/Globdat.h>
#include <jive/model/Names.h>
#include <jive/model/ModelFactory.h>
#include <jive/model/DummyModel.h>
#include <jive/model/DebugModel.h>


JEM_DEFINE_SERIABLE_CLASS( jive::model::DebugModel );


JIVE_BEGIN_PACKAGE( model )


using jem::newInstance;
using jem::System;


//=======================================================================
//   class DebugModel
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  DebugModel::TYPE_NAME = "Debug";


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


DebugModel::DebugModel

  ( const String&      name,
    const Ref<Model>&  child ) :

    Super  (  name ),
    child_ ( child )

{
  if ( ! child_ )
  {
    child_ = newInstance<DummyModel> ( name );
  }

  if ( myName_.size() == 0 )
  {
    myName_ = child_->getName ();
  }

  noiseLevel_ = 1;

  timer_.start ();
}


DebugModel::~DebugModel ()
{}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void DebugModel::readFrom ( ObjectInput& in )
{
  decode ( in, myName_, child_, noiseLevel_ );
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void DebugModel::writeTo ( ObjectOutput& out ) const
{
  encode ( out, myName_, child_, noiseLevel_ );
}


//-----------------------------------------------------------------------
//   findModel
//-----------------------------------------------------------------------


Model* DebugModel::findModel ( const String& name ) const
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


void DebugModel::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  if ( props.contains( myName_ ) )
  {
    Properties  myProps = props.findProps ( myName_ );

    myProps.find ( noiseLevel_, PropNames::NOISE_LEVEL );
  }

  child_->configure ( props, globdat );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void DebugModel::getConfig

  ( const Properties&  props,
    const Properties&  globdat ) const

{
  Properties  myProps = props.makeProps ( myName_ );

  myProps.set ( PropNames::NOISE_LEVEL, noiseLevel_ );

  child_->getConfig ( props, globdat );
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool DebugModel::takeAction

  ( const String&      action,
    const Properties&  params,
    const Properties&  globdat )

{
  using jem::io::endl;
  using jem::io::flush;
  using jem::io::indent;
  using jem::io::outdent;
  using jive::util::Globdat;

  if ( ! out_ )
  {
    out_ = newInstance<Printer_> ( & System::debug( myName_ ) );
  }

  Printer_&  out  = * out_;
  String     name =   child_->getName ();

  bool       result;


  timer_.reset ();

  if ( noiseLevel_ > 0 )
  {
    print ( out, "Model `", name, "\' is taking action `",
            action, "\'" );

    if ( noiseLevel_ > 1 )
    {
      print ( out, " with params:", indent, endl, params );
      print ( out, outdent, endl, endl );
    }
    else
    {
      print ( out, endl, endl );
    }

    print ( out, flush );
  }

  result = child_->takeAction ( action, params, globdat );

  if ( myName_.size() > 0 &&
       Globdat::hasVariable( myName_, globdat ) )
  {
    Properties  vars = Globdat::getVariables ( myName_, globdat );

    vars.set ( VarNames::RUNTIME, timer_.toDouble() );
  }

  if ( noiseLevel_ > 0 )
  {
    if ( result )
    {
      print ( out, "Action completed in ", timer_, '.' );
    }
    else
    {
      print ( out, "No action has been taken." );
    }

    print ( out, endl, endl, flush );
  }

  return result;
}


//-----------------------------------------------------------------------
//   getNoiseLevel
//-----------------------------------------------------------------------


int DebugModel::getNoiseLevel () const
{
  return noiseLevel_;
}


//-----------------------------------------------------------------------
//   setNoiseLevel
//-----------------------------------------------------------------------


void DebugModel::setNoiseLevel ( int level )
{
  noiseLevel_ = level;
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Model> DebugModel::makeNew

  ( const String&      myName,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  String  childName = util::joinNames ( myName, PropNames::MODEL );

  return newInstance<Self> (
    myName,
    newModel ( childName, conf, props, globdat )
  );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void DebugModel::declare ()
{
  ModelFactory::declare ( TYPE_NAME,  & makeNew );
  ModelFactory::declare ( CLASS_NAME, & makeNew );
}


JIVE_END_PACKAGE( model )
