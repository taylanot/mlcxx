
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
#include <jem/base/ClassTemplate.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/Event.h>
#include <jem/util/DictionaryEnumerator.h>
#include <jive/util/error.h>
#include <jive/util/utilities.h>
#include <jive/util/ItemSet.h>
#include <jive/util/Globdat.h>
#include <jive/util/DofSpace.h>
#include <jive/util/Reordering.h>
#include <jive/util/VectorManager.h>


JEM_DEFINE_SERIABLE_CLASS( jive::util::VectorManager );


JIVE_BEGIN_PACKAGE( util )


using jem::newInstance;
using jem::System;


//=======================================================================
//   class VectorManager
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


VectorManager::VectorManager ()
{}


VectorManager::VectorManager

  ( const String&         name,
    const Ref<DofSpace>&  dofs ) :

    myName_ ( name ),
    dofs_   ( dofs )

{
  JEM_PRECHECK2 ( dofs, "NULL DofSpace" );

  connect_ ();
}


VectorManager::~VectorManager ()
{}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void VectorManager::readFrom ( ObjectInput& in )
{
  decode   ( in, myName_, dofs_, data_ );
  connect_ ();
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void VectorManager::writeTo ( ObjectOutput& out ) const
{
  encode ( out, myName_, dofs_, data_ );
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> VectorManager::clone () const
{
  Ref<Self>  cc = newInstance<Self> ( myName_, dofs_ );

  cc->data_ = this->data_.clone ();

  return cc;
}


//-----------------------------------------------------------------------
//   getVector
//-----------------------------------------------------------------------


void VectorManager::getVector

  ( Vector&        vec,
    const String&  name )

{
  const idx_t  dofCount = dofs_->dofCount ();

  if ( data_.find( vec, name ) )
  {
    if ( vec.size() != dofCount )
    {
      resize_ ( vec, dofCount, name );
    }
  }
  else
  {
    vec.resize ( dofCount );

    vec = 0.0;

    data_.set ( name, vec );
  }
}


void VectorManager::getVector

  ( Vector&               vec,
    const String&         name,
    const Ref<DofSpace>&  dofs,
    const Properties&     globdat )

{
  Self::get ( dofs, globdat ) -> getVector ( vec, name );
}


//-----------------------------------------------------------------------
//   findVector
//-----------------------------------------------------------------------


bool VectorManager::findVector

  ( Vector&        vec,
    const String&  name )

{
  if ( ! data_.find( vec, name ) )
  {
    return false;
  }

  const idx_t  dofCount = dofs_->dofCount ();

  if ( vec.size() != dofCount )
  {
    resize_ ( vec, dofCount, name );
  }

  return true;
}


bool VectorManager::findVector

  ( Vector&               vec,
    const String&         name,
    const Ref<DofSpace>&  dofs,
    const Properties&     globdat )

{
  Ref<Self>  vm = Self::find ( dofs, globdat );

  if ( ! vm )
  {
    return false;
  }
  else
  {
    return vm->findVector ( vec, name );
  }
}


//-----------------------------------------------------------------------
//   storeVector
//-----------------------------------------------------------------------


void VectorManager::storeVector

  ( const String&  name,
    const Vector&  vec )

{
  if ( vec.size() != dofs_->dofCount() )
  {
    sizeError ( JEM_FUNC,
                String::format ( "vector `%s\'", name ),
                vec.size(), dofs_->dofCount() );
  }

  data_.set ( name, vec );
}


void VectorManager::storeVector

  ( const String&         name,
    const Vector&         vec,
    const Ref<DofSpace>&  dofs,
    const Properties&     globdat )

{
  Self::get ( dofs, globdat ) -> storeVector ( name, vec );
}


//-----------------------------------------------------------------------
//   store
//-----------------------------------------------------------------------


void VectorManager::store ( const Properties& globdat ) const
{
  Ref<Self>  self = const_cast<Self*> ( this );

  Globdat::storeFor ( Globdat::VECTORS, self, dofs_, globdat );
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


Ref<VectorManager> VectorManager::get

  ( const Ref<DofSpace>&  dofs,
    const Properties&     globdat )

{
  Properties  subdat = Globdat::getDataFor ( dofs, globdat );

  Ref<Self>   vm;

  if ( ! subdat.find( vm, Globdat::VECTORS ) )
  {
    String  name = joinNames ( dofs->getItems()->getName(),
                               "vectorManager" );

    vm = newInstance<Self> ( name, dofs );

    subdat.set ( Globdat::VECTORS, vm );
  }

  return vm;
}


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


Ref<VectorManager> VectorManager::find

  ( const Ref<DofSpace>&  dofs,
    const Properties&     globdat )

{
  Ref<Self>  vm;

  Globdat::findFor ( vm, Globdat::VECTORS, dofs, globdat );

  return vm;
}


//-----------------------------------------------------------------------
//   resize_
//-----------------------------------------------------------------------


void VectorManager::resize_

  ( Vector&        vec,
    idx_t          size,
    const String&  name )

{
  const idx_t  n = vec.size ();

  data_.erase ( name );
  vec.reshape ( size );

  if ( size > n )
  {
    vec[slice(n,END)] = 0.0;
  }

  data_.set ( name, vec );
}


//-----------------------------------------------------------------------
//   connect_
//-----------------------------------------------------------------------


void VectorManager::connect_ ()
{
  using jem::util::connect;

  connect ( dofs_->newSizeEvent,  this, & Self::dofsResized_   );
  connect ( dofs_->newOrderEvent, this, & Self::dofsReordered_ );

  dofs_->resetEvents ();
}


//-----------------------------------------------------------------------
//   dofsResized_
//-----------------------------------------------------------------------


void VectorManager::dofsResized_ ( idx_t dofCount )
{
  if ( data_.size() )
  {
    StringVector  list = data_.listProps ();

    Vector        vec;

    idx_t         n = 0;

    for ( idx_t i = 0; i < list.size(); i++ )
    {
      if ( ! data_.find( vec, list[i] ) )
      {
        continue;
      }

      if ( dofCount < vec.size() )
      {
        if ( myName_.size() )
        {
          print ( System::debug( myName_ ),
                  myName_, " : resizing vector `",
                  list[i], "\' ...\n" );
        }

        resize_ ( vec, dofCount, list[i] );

        n++;
      }
    }

    if ( n > 0 )
    {
      dofs_->resetEvents ();
    }
  }
}


//-----------------------------------------------------------------------
//   dofsReordered_
//-----------------------------------------------------------------------


void VectorManager::dofsReordered_ ( const Reordering& reord )
{
  if ( data_.size() )
  {
    StringVector  list = data_.listProps ();

    Vector        vec;

    for ( idx_t i = 0; i < list.size(); i++ )
    {
      if ( ! data_.find( vec, list[i] ) )
      {
        continue;
      }

      if ( myName_.size() )
      {
        print ( System::debug( myName_ ),
                myName_, " : reordering vector `",
                list[i], "\' ...\n" );
      }

      data_.erase ( list[i] );
      reorder     ( vec, reord, 0.0 );
      data_.set   ( list[i], vec );
    }

    dofs_->resetEvents ();
  }
}


JIVE_END_PACKAGE( util )
