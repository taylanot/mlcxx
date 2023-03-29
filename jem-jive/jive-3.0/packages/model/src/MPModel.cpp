
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
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalInputException.h>
#include <jem/base/IllegalArgumentException.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/mp/Context.h>
#include <jem/util/Pattern.h>
#include <jem/util/Properties.h>
#include <jive/util/utilities.h>
#include <jive/util/XTable.h>
#include <jive/util/ItemSet.h>
#include <jive/util/DofSpace.h>
#include <jive/mp/Globdat.h>
#include <jive/mp/TableExchanger.h>
#include <jive/mp/VectorExchanger.h>
#include <jive/algebra/MPMatrixObject.h>
#include <jive/model/Names.h>
#include <jive/model/Actions.h>
#include <jive/model/Globdat.h>
#include <jive/model/ModelFactory.h>
#include <jive/model/MPModel.h>


JEM_DEFINE_SERIABLE_CLASS( jive::model::MPModel );


JIVE_BEGIN_PACKAGE( model )


using jem::newInstance;
using jem::util::Pattern;
using jive::mp::BorderSet;
using jive::algebra::MPMatrixObj;


//=======================================================================
//   class MPModel
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  MPModel::TYPE_NAME = "MP";


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


MPModel::MPModel ()
{}


MPModel::MPModel

  ( const String&          name,
    const Ref<Model>&      child,
    Ref<TableExchanger>    tex,
    Ref<VectorExchanger>   vex ) :

    Super  ( name  ),
    child_ ( child ),
    tex_   ( tex   ),
    vex_   ( vex   )

{
  JEM_PRECHECK ( child );

  if ( tex_ )
  {
    items_ = tex_->getItems ();
  }

  if ( vex_ )
  {
    dofs_  = vex_->getDofSpace ();

    if      ( ! items_ )
    {
      items_ = dofs_->getItems ();
    }
    else if ( items_ != dofs_->getItems() )
    {
      throw jem::IllegalArgumentException (
        JEM_FUNC,
        "table exchanger and vector exchanger are associated "
        "with different item sets"
      );
    }
  }
}


MPModel::MPModel

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat ) :

    Super ( name )

{
  using jive::util::joinNames;

  Properties  myConf    = conf .makeProps ( myName_ );
  Properties  myProps   = props.findProps ( myName_ );

  String      myContext = getContext ();

  String      path;


  child_ = newModel ( joinNames ( myName_, PropNames::MODEL ),
                      conf, props, globdat );

  if ( myProps.find( path, PropNames::ITEMS ) )
  {
    items_ = ItemSet ::get  ( path,   globdat, myContext );
    dofs_  = DofSpace::find ( items_, globdat );
  }
  else
  {
    dofs_  = DofSpace::get        ( globdat, myContext );
    items_ = dofs_ ->getItems     ();
    path   = items_->getItemsName ();
  }

  myConf.set ( PropNames::ITEMS, path );

  if ( BorderSet::hasBorders( items_, globdat ) )
  {
    tex_ = TableExchanger::get ( items_, globdat );

    if ( dofs_ )
    {
      vex_ = VectorExchanger::get ( dofs_, globdat );
    }
  }

  setDefaults ();
}


MPModel::~MPModel ()
{}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void MPModel::readFrom ( ObjectInput& in )
{
  decode ( in,
           myName_,
           items_,
           dofs_,
           child_ );

  decode ( in,
           tabNames_,
           vecActions_,
           vecNames_ );
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void MPModel::writeTo ( ObjectOutput& out ) const
{
  encode ( out,
           myName_,
           items_,
           dofs_,
           child_ );

  encode ( out,
           tabNames_,
           vecActions_,
           vecNames_ );
}


//-----------------------------------------------------------------------
//   findModel
//-----------------------------------------------------------------------


Model* MPModel::findModel ( const String& name ) const
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


void MPModel::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  using jem::IllegalInputException;

  if ( props.contains( myName_ ) )
  {
    Properties    myProps = props.findProps ( myName_ );

    StringVector  svec;

    if ( myProps.find( svec, PropNames::TABLES ) )
    {
      idx_t  n = svec.size ();

      tabNames_.clear   ();
      tabNames_.reserve ( n );

      try
      {
        for ( idx_t i = 0; i < n; i++ )
        {
          addTable ( svec[i] );
        }
      }
      catch ( IllegalInputException& ex )
      {
        ex.setContext ( myProps.getContext( PropNames::TABLES ) );
        throw;
      }
    }

    if ( myProps.find( svec, PropNames::VECTORS ) )
    {
      idx_t  n = svec.size ();

      if ( n % 2 != 0 )
      {
        props.propertyError (
          PropNames::VECTORS,
          "array size must be even (it must contain pairs of "
          "action name, vector name)"
        );
      }

      vecActions_.clear   ();
      vecNames_  .clear   ();
      vecActions_.reserve ( n / 2 );
      vecNames_  .reserve ( n / 2 );

      try
      {
        for ( idx_t i = 0; i < n; i += 2 )
        {
          addVector ( svec[i], svec[i + 1] );
        }
      }
      catch ( IllegalInputException& ex )
      {
        ex.setContext ( myProps.getContext( PropNames::VECTORS ) );
        throw;
      }
    }
  }

  child_->configure ( props, globdat );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void MPModel::getConfig

  ( const Properties&  props,
    const Properties&  globdat ) const

{
  Properties    myProps = props.makeProps ( myName_ );

  StringVector  tables  (     tabNames_.size() );
  StringVector  vectors ( 2 * vecNames_.size() );


  for ( idx_t i = 0; i < tabNames_.size(); i++ )
  {
    tables[i] = tabNames_[i];
  }

  for ( idx_t i = 0; i < vecNames_.size(); i++ )
  {
    vectors[2 * i + 0] = vecActions_[i];
    vectors[2 * i + 1] = vecNames_  [i];
  }

  myProps.set ( PropNames::TABLES,  tables  );
  myProps.set ( PropNames::VECTORS, vectors );

  child_->getConfig ( props, globdat );
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool MPModel::takeAction

  ( const String&      action,
    const Properties&  params,
    const Properties&  globdat )

{
  using jive::util::XTable;

  bool    result;
  idx_t   n;

  if ( action == Actions::INIT )
  {
    if ( tex_ == nullptr && items_ &&
         BorderSet::hasBorders( items_, globdat ) )
    {
      tex_ = TableExchanger::get ( items_, globdat );
    }

    if ( vex_ == nullptr && dofs_ &&
         BorderSet::hasBorders( items_, globdat ) )
    {
      vex_ = VectorExchanger::get ( dofs_, globdat );
    }

    child_->takeAction ( action, params, globdat );

    return true;
  }

  if ( tex_ && action == Actions::GET_TABLE )
  {
    result = child_->takeAction ( action, params, globdat );

    if ( result )
    {
      Ref<XTable>  table;
      Vector       weights;
      String       name;

      params.get ( table, ActionParams::TABLE );

      if ( table->getRowItems() != items_ )
      {
        return result;
      }

      params.get ( name, ActionParams::TABLE_NAME );

      n = tabNames_.size ();

      for ( idx_t i = 0; i < n; i++ )
      {
        if ( Pattern::matches( tabNames_[i], name ) )
        {
          if ( params.find( weights, ActionParams::TABLE_WEIGHTS ) )
          {
            tex_->exchange ( *table, weights );
          }
          else
          {
            tex_->exchange ( *table );
          }

          break;
        }
      }
    }

    return result;
  }

  if ( action == Actions::GET_MP_MODEL )
  {
    params.set ( ActionParams::MP_MODEL, this );

    return true;
  }

  if ( ! vex_ )
  {
    return child_->takeAction ( action, params, globdat );
  }

  n = vecNames_.size ();

  for ( idx_t i = 0; i < n; i++ )
  {
    if ( Pattern::matches( vecActions_[i], action ) )
    {
      Vector  vec;

      result = child_->takeAction ( action, params, globdat );

      if ( result && params.find( vec, vecNames_[i] ) )
      {
        vex_->exchange ( vec );
      }

      return result;
    }
  }

  for ( int i = 0; i < 3; i++ )
  {
    if ( action == Actions::NEW_MATRIX[i] )
    {
      Ref<AbstractMatrix>  mat;

      result = child_->takeAction ( action, params, globdat );

      if ( result && params.find( mat, ActionParams::MATRIX[i] ) )
      {
        if ( ! mat->isDistributed() )
        {
          mat = newInstance<MPMatrixObj> ( mat, vex_ );

          params.set ( ActionParams::MATRIX[i], mat );
        }
      }

      return result;
    }
  }

  return child_->takeAction ( action, params, globdat );
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void MPModel::clear ()
{
  tabNames_  .clear ();
  vecActions_.clear ();
  vecNames_  .clear ();
}


//-----------------------------------------------------------------------
//   addTable
//-----------------------------------------------------------------------


void MPModel::addTable ( const String& tname )
{
  Pattern ::check    ( tname );
  tabNames_.pushBack ( tname );
}


//-----------------------------------------------------------------------
//   eraseTable
//-----------------------------------------------------------------------


bool MPModel::eraseTable ( const String& tname )
{
  bool   result = false;
  idx_t  i;


  i = 0;

  while ( i < tabNames_.size() )
  {
    if ( tabNames_[i] == tname )
    {
      result = true;

      tabNames_[i].swap ( tabNames_.back() );
      tabNames_.popBack ();
    }
    else
    {
      i++;
    }
  }

  return result;
}


//-----------------------------------------------------------------------
//   addVector
//-----------------------------------------------------------------------


void MPModel::addVector

  ( const String&  action,
    const String&  vname )

{
  Pattern   ::check    ( action );
  vecActions_.pushBack ( action );
  vecNames_  .pushBack ( vname  );
}


//-----------------------------------------------------------------------
//   eraseVector
//-----------------------------------------------------------------------


bool MPModel::eraseVector ( const String& action )
{
  bool   result = false;
  idx_t  i;


  i = 0;

  while ( i < vecActions_.size() )
  {
    if ( vecActions_[i] == action )
    {
      result = true;

      vecActions_[i].swap ( vecActions_.back() );
      vecNames_  [i].swap ( vecNames_  .back() );

      vecActions_.popBack ();
      vecNames_  .popBack ();
    }
    else
    {
      i++;
    }
  }

  return result;
}


//-----------------------------------------------------------------------
//   setDefaults
//-----------------------------------------------------------------------


void MPModel::setDefaults ()
{
  tabNames_  .resize ( 1 );
  vecActions_.resize ( 4 );
  vecNames_  .resize ( 4 );

  tabNames_  [0] = "*";

  vecActions_[0] = Actions::GET_INT_VECTOR;
  vecActions_[1] = Actions::GET_EXT_VECTOR;
  vecActions_[2] = Actions::UPD_MATRIX0;
  vecActions_[3] = Actions::GET_MATRIX0;

  vecNames_  [0] = ActionParams::INT_VECTOR;
  vecNames_  [1] = ActionParams::EXT_VECTOR;
  vecNames_  [2] = ActionParams::INT_VECTOR;
  vecNames_  [3] = ActionParams::INT_VECTOR;
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Model> MPModel::makeNew

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


void MPModel::declare ()
{
  ModelFactory::declare ( TYPE_NAME,  & makeNew );
  ModelFactory::declare ( CLASS_NAME, & makeNew );
}


JIVE_END_PACKAGE( model )
