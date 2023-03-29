
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
#include <jem/util/Event.h>
#include <jem/util/Properties.h>
#include <jive/util/utilities.h>
#include <jive/util/ItemSet.h>
#include <jive/util/DofSpace.h>
#include <jive/util/VectorManager.h>
#include <jive/gl/Names.h>
#include <jive/gl/DataSetFactory.h>
#include <jive/gl/VectorData.h>


JEM_DEFINE_CLASS( jive::gl::VectorData );


JIVE_BEGIN_PACKAGE( gl )


using jem::newInstance;
using jem::IllegalInputException;


//=======================================================================
//   class VectorData
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  VectorData::TYPE_NAME = "Vector";


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


VectorData::VectorData

  ( const String&  name,
    const String&  vpath ) :

    Super ( name )

{
  nullValue_ = 0.0;

  setPath_ ( vpath, AppContext() );
}


VectorData::VectorData

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props ) :

    Super ( name )

{
  Properties  myConf  = conf.makeProps ( myName_ );
  Properties  myProps = props.getProps ( myName_ );

  String      vpath;

  idx_t       i;


  i = myName_.rfind ( '.' );

  if ( i >= 0 )
  {
    vpath = myName_[slice(i + 1,END)];
  }
  else
  {
    vpath = myName_;
  }

  if ( ! vpath.size() )
  {
    myProps.get  ( vpath, PropNames::VECTOR );
  }
  else
  {
    myProps.find ( vpath, PropNames::VECTOR );
  }

  nullValue_ = 0.0;

  myConf.set ( PropNames::VECTOR, vpath );
  setPath_   ( vpath, AppContext() );
}


VectorData::~VectorData ()
{}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void VectorData::update

  ( const Properties&  globdat,
    const AppContext&  apx )

{
  using jive::util::expandString;
  using jive::util::VectorManager;


  if ( ! dofs_[apx] )
  {
    getDofSpace_ ( globdat, apx );
  }

  String  vname = expandString              ( vname_,     globdat );
  bool    found = VectorManager::findVector ( vdata_,     vname,
                                              dofs_[apx], globdat );

  if ( ! found )
  {
    String  vpath = expandString ( vpath_, globdat );

    throw IllegalInputException (
      getContext     (),
      String::format ( "undefined vector: `%s\'", vpath )
    );

    return;
  }

  updated_ = false;

  dofs_[apx]->resetEvents ();

  updated_ = true;
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void VectorData::configure

  ( const Properties&  props,
    const Properties&  globdat,
    const AppContext&  apx )

{
  if ( props.contains( myName_ ) )
  {
    Properties  myProps = props.findProps ( myName_ );

    String      vpath;


    myProps.find ( nullValue_, PropNames::NULL_VALUE );

    if ( myProps.find( vpath, PropNames::VECTOR )
         && vpath != vpath_ )
    {
      try
      {
        setPath_ ( vpath, apx );
      }
      catch ( IllegalInputException& ex )
      {
        ex.setContext ( myProps.getContext( PropNames::VECTOR ) );
        throw;
      }
    }
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void VectorData::getConfig ( const Properties& props ) const
{
  Properties  myProps = props.makeProps ( myName_ );

  myProps.set ( PropNames::VECTOR,     vpath_     );
  myProps.set ( PropNames::NULL_VALUE, nullValue_ );
}


//-----------------------------------------------------------------------
//   rowCount
//-----------------------------------------------------------------------


idx_t VectorData::rowCount ( const AppContext& apx ) const
{
  if ( updated_ )
  {
    return dofs_[apx]->itemCount ();
  }
  else
  {
    return 0;
  }
}


//-----------------------------------------------------------------------
//   columnCount
//-----------------------------------------------------------------------


idx_t VectorData::columnCount ( const AppContext& apx ) const
{
  if ( updated_ )
  {
    return dofs_[apx]->typeCount ();
  }
  else
  {
    return 0;
  }
}


//-----------------------------------------------------------------------
//   findColumn
//-----------------------------------------------------------------------


idx_t VectorData::findColumn

  ( const String&      name,
    const AppContext&  apx ) const

{
  if ( updated_ )
  {
    return dofs_[apx]->findType ( name );
  }
  else
  {
    return -1;
  }
}


//-----------------------------------------------------------------------
//   getItems
//-----------------------------------------------------------------------


ItemSet* VectorData::getItems ( const AppContext& apx ) const
{
  if ( ! dofs_[apx] )
  {
    return nullptr;
  }
  else
  {
    return dofs_[apx]->getItems ();
  }
}


//-----------------------------------------------------------------------
//   getData
//-----------------------------------------------------------------------


void VectorData::getData

  ( const Matrix&      buf,
    const IdxVector&   irows,
    const IdxVector&   jcols,
    const AppContext&  apx ) const

{
  JEM_PRECHECK ( buf.size(0) == irows.size() &&
                 buf.size(1) == jcols.size() );

  if ( ! updated_ )
  {
    buf = nullValue_;
    return;
  }

  const idx_t  m     = buf.size (0);
  const idx_t  n     = buf.size (1);

  IdxVector    idofs ( m * n );


  idofs = -1;

  dofs_[apx]->findDofIndices ( idofs, irows, jcols );

  for ( idx_t j = 0; j < n; j++ )
  {
    for ( idx_t i = 0; i < m; i++ )
    {
      idx_t  k = idofs[i * n + j];

      if ( k < 0 )
      {
        buf(i,j) = nullValue_;
      }
      else
      {
        buf(i,j) = vdata_[k];
      }
    }
  }
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<DataSet> VectorData::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat,
    const AppContext&  apx )

{
  return newInstance<Self> ( name, conf, props );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void VectorData::declare ()
{
  DataSetFactory::declare ( TYPE_NAME,  & makeNew );
  DataSetFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   setPath_
//-----------------------------------------------------------------------


void VectorData::setPath_

  ( const String&      vpath,
    const AppContext&  apx )

{
  using jem::util::disconnect;

  String  vname;

  idx_t   i = vpath.find ( '/' );


  if ( i < 0 )
  {
    vname = vpath;
  }
  else
  {
    vname = vpath[slice(i + 1,END)].stripWhite ();
  }

  if ( i == 0 || ! vname.size() )
  {
    throw IllegalInputException (
      getContext (),
      String::format ( "invalid vector name: `%s\'", vpath )
    );
  }

  vpath_   = vpath;
  vname_   = vname;
  updated_ = false;

  if ( dofs_[apx] )
  {
    disconnect ( dofs_[apx]->newSizeEvent,
                 this, & Self::invalidate_ );

    disconnect ( dofs_[apx]->newOrderEvent,
                 this, & Self::invalidate_ );

    dofs_[apx] = nullptr;
  }
}


//-----------------------------------------------------------------------
//   getDofSpace_
//-----------------------------------------------------------------------


void VectorData::getDofSpace_

  ( const Properties&  globdat,
    const AppContext&  apx )

{
  using jem::util::connect;
  using jem::util::disconnect;

  String  context = getContext ();

  idx_t   i;


  updated_ = false;

  if ( dofs_[apx] )
  {
    disconnect ( dofs_[apx]->newSizeEvent,
                 this, & Self::invalidate_ );

    disconnect ( dofs_[apx]->newOrderEvent,
                 this, & Self::invalidate_ );

    dofs_[apx] = nullptr;
  }

  i = vpath_.find ( '/' );

  if ( i < 0 )
  {
    dofs_[apx] = DofSpace::find ( globdat );

    if ( ! dofs_[apx] )
    {
      throw IllegalInputException (
        context,
        "no item set has been specified and no primary "
        "DOF space has been set"
      );
    }
  }
  else
  {
    String        path  = vpath_[slice(BEGIN,i)].stripWhite ();

    Ref<ItemSet>  items = ItemSet::get ( path, globdat, context );

    dofs_[apx] = DofSpace::get ( items, globdat, context );
  }

  connect ( dofs_[apx]->newSizeEvent,  this, & Self::invalidate_ );
  connect ( dofs_[apx]->newOrderEvent, this, & Self::invalidate_ );
}


//-----------------------------------------------------------------------
//   invalidate_
//-----------------------------------------------------------------------


void VectorData::invalidate_ ()
{
  updated_ = false;
}


JIVE_END_PACKAGE( gl )
