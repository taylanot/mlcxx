
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
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jive/model/ModelFactory.h>
#include <jive/model/DummyModel.h>


JEM_DEFINE_SERIABLE_CLASS( jive::model::DummyModel );


JIVE_BEGIN_PACKAGE( model )


//=======================================================================
//   class DummyModel
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  DummyModel::TYPE_NAME = "None";


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


DummyModel::DummyModel ( const String& name ) : Super ( name )
{}


DummyModel::~DummyModel ()
{}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void DummyModel::readFrom ( ObjectInput& in )
{
  decode ( in, myName_ );
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void DummyModel::writeTo ( ObjectOutput& out ) const
{
  encode ( out, myName_ );
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool DummyModel::takeAction

  ( const String&      action,
    const Properties&  params,
    const Properties&  globdat )

{
  return false;
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Model> DummyModel::makeNew

  ( const String&      myName,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  return jem::newInstance<Self> ( myName );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void DummyModel::declare ()
{
  ModelFactory::declare ( TYPE_NAME,  & makeNew );
  ModelFactory::declare ( CLASS_NAME, & makeNew );
}


JIVE_END_PACKAGE( model )
