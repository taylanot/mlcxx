
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
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/Flex.h>
#include <jem/util/Pattern.h>
#include <jem/util/Properties.h>
#include <jive/Array.h>
#include <jive/util/utilities.h>
#include <jive/model/Names.h>
#include <jive/model/ModelFactory.h>
#include <jive/model/ActionException.h>
#include <jive/model/FilterModel.h>


JEM_DEFINE_SERIABLE_CLASS( jive::model::FilterModel );


JIVE_BEGIN_PACKAGE( model )


using jem::IllegalInputException;
using jem::util::Pattern;


//=======================================================================
//   class FilterModel
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  FilterModel::TYPE_NAME = "Filter";


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


FilterModel::FilterModel ()
{}


FilterModel::FilterModel

  ( const String&      name,
    const Ref<Model>&  child ) :

    Super  (  name ),
    child_ ( child )

{
  JEM_PRECHECK ( child );

  if ( myName_.size() == 0 )
  {
    myName_ = child->getName ();
  }
}


FilterModel::~FilterModel ()
{}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void FilterModel::readFrom ( ObjectInput& in )
{
  decode ( in, myName_, child_, patterns_, actions_ );
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void FilterModel::writeTo ( ObjectOutput& out ) const
{
  encode ( out, myName_, child_, patterns_, actions_ );
}


//-----------------------------------------------------------------------
//   findModel
//-----------------------------------------------------------------------


Model* FilterModel::findModel ( const String& name ) const
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


void FilterModel::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  if ( props.contains( myName_ ) )
  {
    Properties    myProps = props.findProps ( myName_ );

    StringVector  filters;

    if ( myProps.find( filters, PropNames::FILTERS ) )
    {
      idx_t  n = filters.size ();

      if ( n % 2 != 0 )
      {
        myProps.propertyError (
          PropNames::FILTERS,
          "length of filter array must be even"
        );
      }

      patterns_.clear   ();
      actions_ .clear   ();
      patterns_.reserve ( n / 2 );
      actions_ .reserve ( n / 2 );

      try
      {
        for ( idx_t i = 0; i < n; i += 2 )
        {
          addFilter ( filters[i], filters[i + 1] );
        }
      }
      catch ( IllegalInputException& ex )
      {
        ex.setContext ( myProps.getContext( PropNames::FILTERS ) );
        throw;
      }
    }
  }

  child_->configure ( props, globdat );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void FilterModel::getConfig

  ( const Properties&  props,
    const Properties&  globdat ) const

{
  Properties    myProps = props.makeProps ( myName_ );

  const idx_t   n       = actions_.size   ();

  StringVector  filters ( 2 * n );


  for ( idx_t i = 0; i < n; i++ )
  {
    String  a = actions_[i];

    if ( a.size() > 0 && a[0] == '%' )
    {
      if      ( a[1] == 'd' )
      {
        a = "%drop";
      }
      else if ( a[1] == 'p' )
      {
        a = "%pass";
      }
      else if ( a[1] == 'e' )
      {
        a = "%error";
      }
    }

    filters[2 * i + 0] = patterns_[i];
    filters[2 * i + 1] = a;
  }

  myProps.set ( PropNames::FILTERS, filters );

  child_->getConfig ( props, globdat );
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool FilterModel::takeAction

  ( const String&      action,
    const Properties&  params,
    const Properties&  globdat )

{
  const idx_t  n = patterns_.size ();

  for ( idx_t i = 0; i < n; i++ )
  {
    if ( Pattern::matches( patterns_[i], action ) )
    {
      const String& a = actions_[i];

      if ( a[0] == '%' )
      {
        if ( a[1] == '%' )
        {
          return child_->takeAction ( a[slice(1,END)],
                                      params, globdat );
        }

        if ( a[1] == 'd' )
        {
          return false;
        }

        if ( a[1] == 'p' )
        {
          return child_->takeAction ( action, params, globdat );
        }

        if ( a[1] == 'e' )
        {
          throw ActionException (
            myName_, action, "illegal action"
          );
        }
      }

      return child_->takeAction ( a, params, globdat );
    }
  }

  return false;
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void FilterModel::clear ()
{
  patterns_.clear ();
  actions_ .clear ();
}


//-----------------------------------------------------------------------
//   addFilter
//-----------------------------------------------------------------------


void FilterModel::addFilter

  ( const String&  pattern,
    const String&  action )

{
  const idx_t  n = patterns_.size ();

  Pattern::check ( pattern );

  String  a = processAction_ ( action );

  for ( idx_t i = 0; i < n; i++ )
  {
    if ( pattern == patterns_[i] )
    {
      actions_[i] = a;
      return;
    }
  }

  patterns_.pushBack ( pattern );
  actions_ .pushBack ( std::move( a ) );
}


//-----------------------------------------------------------------------
//   eraseFilter
//-----------------------------------------------------------------------


bool FilterModel::eraseFilter ( const String& pattern )
{
  const idx_t  n = patterns_.size ();


  for ( idx_t i = 0; i < n; i++ )
  {
    if ( pattern == patterns_[i] )
    {
      if ( i < n - 1 )
      {
        patterns_[i] = patterns_.back ();
        actions_ [i] = actions_ .back ();
      }

      patterns_.popBack ();
      actions_ .popBack ();

      return true;
    }
  }

  return false;
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Model> FilterModel::makeNew

  ( const String&      myName,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  String  childName = util::joinNames ( myName, PropNames::MODEL );

  return jem::newInstance<Self> (
    myName,
    newModel ( childName, conf, props, globdat )
  );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void FilterModel::declare ()
{
  ModelFactory::declare ( TYPE_NAME,  & makeNew );
  ModelFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   processAction_
//-----------------------------------------------------------------------


String FilterModel::processAction_ ( const String& action )
{
  if ( action.size() == 0 )
  {
    return String ( "%d" );
  }

  if ( action[0] == '%' )
  {
    String cmd = action[slice(1,END)];

    if ( cmd.size() > 0 && cmd[0] == '%' )
    {
      return action;
    }

    cmd = cmd.toLower ();

    if ( cmd == "drop" )
    {
      return String ( "%d" );
    }

    if ( cmd == "pass" )
    {
      return String ( "%p" );
    }

    if ( cmd == "error" )
    {
      return String ( "%e" );
    }
  }
  else
  {
    return action;
  }

  throw IllegalInputException (
    CLASS_NAME,
    String::format ( "invalid action: `%s\'", action )
  );

  return String ();
}


JIVE_END_PACKAGE( model )
