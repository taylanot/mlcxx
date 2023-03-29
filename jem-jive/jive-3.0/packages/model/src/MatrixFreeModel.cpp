
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
#include <jem/base/Float.h>
#include <jem/base/ClassTemplate.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/Event.h>
#include <jive/util/utilities.h>
#include <jive/util/ItemSet.h>
#include <jive/util/DofSpace.h>
#include <jive/model/Names.h>
#include <jive/model/Actions.h>
#include <jive/model/ModelFactory.h>
#include <jive/model/ImplicitMatrix.h>
#include <jive/model/MatrixFreeModel.h>


JEM_DEFINE_SERIABLE_CLASS( jive::model::MatrixFreeModel );


JIVE_BEGIN_PACKAGE( model )


using jem::newInstance;
using jive::util::joinNames;


//=======================================================================
//   private functions
//=======================================================================

//-----------------------------------------------------------------------
//   newValuesHandler_
//-----------------------------------------------------------------------


void newValuesHandler_ ( AbstractMatrix& matrix )
{
  matrix.newValuesEvent.emit ( matrix );
}


//=======================================================================
//   class MatrixFreeModel
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  MatrixFreeModel::TYPE_NAME = "MatrixFree";


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


MatrixFreeModel::MatrixFreeModel ()
{}


MatrixFreeModel::MatrixFreeModel

  ( const String&         name,
    const Ref<DofSpace>&  dofs,
    const Ref<Model>&     child ) :

    Super  ( name  ),
    dofs_  ( dofs  ),
    child_ ( child )

{
  JEM_PRECHECK ( dofs && child );

  eps_       = jem::Float::EPSILON;
  symmetric_ = false;
}


MatrixFreeModel::MatrixFreeModel

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat ) :

    Super ( name  )

{
  // First create the child model, and then get the DOF space.

  child_ = newModel ( joinNames ( myName_, PropNames::MODEL ),
                      conf,
                      props,
                      globdat );

  dofs_      = DofSpace::get ( globdat, getContext() );
  eps_       = jem::Float::EPSILON;
  symmetric_ = false;
}


MatrixFreeModel::~MatrixFreeModel ()
{}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void MatrixFreeModel::readFrom ( ObjectInput& in )
{
  decode ( in,
           myName_,
           eps_,
           symmetric_,
           dofs_,
           child_ );

  matrix_ = nullptr;
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void MatrixFreeModel::writeTo ( ObjectOutput& out ) const
{
  encode ( out,
           myName_,
           eps_,
           symmetric_,
           dofs_,
           child_ );
}


//-----------------------------------------------------------------------
//   findModel
//-----------------------------------------------------------------------


Model* MatrixFreeModel::findModel ( const String& name ) const
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


void MatrixFreeModel::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  if ( props.contains( myName_ ) )
  {
    Properties  myProps = props.findProps ( myName_ );

    if ( myProps.find( eps_, PropNames::EPSILON, 1.0e-20, 1.0e20 ) &&
         matrix_ )
    {
      matrix_->setEpsilon ( eps_ );
    }

    myProps.find ( symmetric_, PropNames::SYMMETRIC );
  }

  child_->configure ( props, globdat );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void MatrixFreeModel::getConfig

  ( const Properties&  props,
    const Properties&  globdat ) const

{
  Properties  myProps = props.makeProps ( myName_ );

  myProps.set ( PropNames::EPSILON,   eps_       );
  myProps.set ( PropNames::SYMMETRIC, symmetric_ );

  child_->getConfig ( props, globdat );
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool MatrixFreeModel::takeAction

  ( const String&      action,
    const Properties&  params,
    const Properties&  globdat )

{
  if ( matrix_ )
  {
    return matrix_->takeAction ( action, params, globdat );
  }

  if ( action == Actions::NEW_MATRIX0 )
  {
    int  traits = 0;

    if ( symmetric_ )
    {
      traits |= ImplicitMatrix::SYMMETRIC;
    }

    matrix_ = newInstance<ImplicitMatrix>

      ( joinNames ( myName_, PropNames::MATRIX ),
        child_,  dofs_,
        globdat, traits );

    params.set ( ActionParams::MATRIX0, matrix_ );

    return true;
  }

  return child_->takeAction ( action, params, globdat );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Model> MatrixFreeModel::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  return newInstance<Self> ( name, conf, props, globdat );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void MatrixFreeModel::declare ()
{
  ModelFactory::declare ( TYPE_NAME,  & Self::makeNew );
  ModelFactory::declare ( CLASS_NAME, & Self::makeNew );
}


JIVE_END_PACKAGE( model )
