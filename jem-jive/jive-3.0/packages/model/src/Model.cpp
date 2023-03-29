
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


#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalInputException.h>
#include <jem/util/Properties.h>
#include <jive/util/utilities.h>
#include <jive/model/Globdat.h>
#include <jive/model/Model.h>


JEM_DEFINE_CLASS( jive::model::Model );


JIVE_BEGIN_PACKAGE( model )


//=======================================================================
//   class Model
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


Model::Model ( const String& name ) :

  Super ( name )

{}


Model::Model ( const Self& rhs ) :

  Super ( rhs.myName_ )

{}


Model::~Model ()
{}


//-----------------------------------------------------------------------
//   findModel
//-----------------------------------------------------------------------


Model* Model::findModel ( const String& name ) const
{
  if ( name == myName_ )
  {
    return const_cast<Model*> ( this );
  }
  else
  {
    return 0;
  }
}


//-----------------------------------------------------------------------
//   configure & getConfig
//-----------------------------------------------------------------------


void Model::configure

  ( const Properties&  props,
    const Properties&  globdat )

{}


void Model::getConfig

  ( const Properties&  props,
    const Properties&  globdat ) const

{}


//-----------------------------------------------------------------------
//   getContext
//-----------------------------------------------------------------------


String Model::getContext () const
{
  return NamedObject::makeContext ( "model", myName_ );
}


//-----------------------------------------------------------------------
//   store
//-----------------------------------------------------------------------


void Model::store ( const Properties& globdat ) const
{
  Ref<Self> self ( const_cast<Self*>( this ) );

  globdat.set ( Globdat::MODEL, self );
}


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


Ref<Model> Model::find ( const Properties& globdat )
{
  Ref<Model> model;

  globdat.find ( model, Globdat::MODEL );

  return model;
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


Ref<Model> Model::get

  ( const Properties&  globdat,
    const String&      context )

{
  Ref<Model> model;

  if ( ! globdat.find( model, Globdat::MODEL ) )
  {
    throw jem::IllegalInputException (
      context,
      "no model has been defined"
    );
  }

  return model;
}


//-----------------------------------------------------------------------
//   extract
//-----------------------------------------------------------------------


Ref<Model> Model::extract ( const Properties& globdat )
{
  Ref<Model> model;

  if ( globdat.find( model, Globdat::MODEL ) )
  {
    globdat.erase ( Globdat::MODEL );
  }

  return model;
}


JIVE_END_PACKAGE( model )
