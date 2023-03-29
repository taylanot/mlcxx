
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
#include <jem/base/IllegalInputException.h>
#include <jem/mp/Context.h>
#include <jem/util/Flex.h>
#include <jem/util/Properties.h>
#include <jem/util/Dictionary.h>
#include <jem/util/DictionaryEnumerator.h>
#include <jem/xml/utilities.h>
#include <jem/xml/Tokenizer.h>
#include <jive/util/ItemSet.h>
#include <jive/util/ArrayItemGroup.h>
#include <jive/util/ParserActions.h>
#include <jive/util/ItemGroupParser.h>
#include <jive/mp/Globdat.h>
#include <jive/app/Names.h>
#include <jive/app/ModuleFactory.h>
#include <jive/app/GroupgenModule.h>


JEM_DEFINE_CLASS( jive::app::GroupgenModule );


JIVE_BEGIN_PACKAGE( app )


using jem::newInstance;
using jem::System;
using jem::util::Dictionary;
using jem::xml::Tokenizer;
using jive::util::ItemSet;
using jive::util::ItemGroupParser;


//=======================================================================
//   class GroupgenModule::Work_
//=======================================================================


class GroupgenModule::Work_
{
 public:

  explicit inline         Work_

    ( const String&         context );

  void                    initParser

    ( const Properties&     globdat );

  idx_t                   makeGroups

    ( const String&         myName,
      const Properties&     conf,
      const Properties&     props );


 public:

  const String            context;
  String                  itemName;

  Ref<ItemSet>            items;
  Ref<Dictionary>         groups;

  Ref<Tokenizer>          tokzer;
  Ref<ItemGroupParser>    parser;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline GroupgenModule::Work_::Work_ ( const String& ctx ) :

  context ( ctx )

{}


//-----------------------------------------------------------------------
//   initParser
//-----------------------------------------------------------------------


void GroupgenModule::Work_::initParser

  ( const Properties& globdat )

{
  using jive::util::ParserActions;
  using jive::util::ParserOptions;

  tokzer = newInstance<Tokenizer>       ();
  parser = newInstance<ItemGroupParser> ();

  if ( mp::Globdat::getMPContext( globdat )->size() > 1 )
  {
    Properties  params;

    params.set ( ParserOptions::SKIP_UNKNOWN, true );

    parser->takeAction ( ParserActions::SET_OPTIONS, params );
  }
}


//-----------------------------------------------------------------------
//   makeGroups
//-----------------------------------------------------------------------


idx_t GroupgenModule::Work_::makeGroups

  ( const String&      myName,
    const Properties&  conf,
    const Properties&  props )

{
  using jem::IllegalInputException;
  using jem::io::Writer;
  using jem::util::Flex;
  using jem::util::DictEnum;
  using jive::util::ItemGroup;
  using jive::util::XItemGroup;
  using jive::util::ArrayItemGroup;

  Writer&           info   = System::info ( myName );

  Properties        gconf  = conf;
  Properties        gprops = props;

  Flex<String>      tempGroups;

  Ref<DictEnum>     e;
  Ref<ItemGroup>    g;
  Ref<XItemGroup>   xg;

  String            name;
  String            expr;

  bool              temp;
  idx_t             count;
  idx_t             i, n;


  count = 0;

  for ( e = props.enumerate(); ! e->atEnd(); e->toNext() )
  {
    name   = e->getKey ();
    gconf  = conf .makeProps ( name );
    gprops = props.getProps  ( name );
    temp   = false;

    gprops.find ( name, PropNames::NAME );
    gprops.get  ( expr, PropNames::EXPR );
    gprops.find ( temp, PropNames::TEMP );

    gconf .set  ( PropNames::NAME, name );
    gconf .set  ( PropNames::EXPR, expr );
    gconf .set  ( PropNames::TEMP, temp );

    if ( groups->contains( name ) )
    {
      throw IllegalInputException (
        context,
        String::format (
          "%s group `%s\' already exists",
          & itemName,
          & name
        )
      );
    }

    if ( temp )
    {
      tempGroups.pushBack ( name );
    }

    g = jem::dynamicCast<ItemGroup> ( groups->get( expr ) );

    if ( ! g || g->getItems() != items.get() )
    {
      xg = newInstance<ArrayItemGroup> ( items );

      tokzer->setInput ( expr );
      parser->setGroup ( xg );

      try
      {
        parser->parseInput ( tokzer, "expression" );
      }
      catch ( IllegalInputException& ex )
      {
        String  what = ex.what ();

        ex.setContext ( context );
        ex.setMessage (
          String::format (
            "error creating %s group `%s\': %s",
            itemName,
            name,
            what
          )
        );

        throw;
      }

      g = xg;
    }

    groups->insert ( name, g );
    count++;

    print ( info, "  created group `", name, "\'\n" );
  }

  for ( i = 0, n = tempGroups.size(); i < n; i++ )
  {
    groups->erase ( tempGroups[i] );

    print ( info, "  deleted group `", tempGroups[i], "\'\n" );
  }

  return (count - n );
}


//=======================================================================
//   class GroupgenModule
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  GroupgenModule::TYPE_NAME = "Groupgen";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


GroupgenModule::GroupgenModule ( const String& name ) :

    Super ( name )

{}


GroupgenModule::~GroupgenModule ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


Module::Status GroupgenModule::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  using jem::io::Writer;
  using jem::xml::makeTag;
  using jive::util::ItemGroup;


  Writer&       out     = System ::out     ();

  StringVector  list    = ItemSet::listAll ( globdat );

  Properties    myConf  = conf .makeProps  ( myName_ );
  Properties    myProps = props.findProps  ( myName_ );

  Work_         w       ( getContext() );

  String        propName;

  idx_t         i, k, n;


  for ( i = 0, n = list.size(); i < n; i++ )
  {
    globdat.get ( w.items, list[i] );

    w.itemName = w.items->getItemName ();
    propName   = makeTag ( w.itemName, false ) + "Groups";

    if ( myProps.contains( propName ) )
    {
      print ( out, "Generating ", w.itemName, " groups ...\n" );

      if ( ! w.parser )
      {
        w.initParser ( globdat );
      }

      w.groups = ItemGroup::getFor ( w.items, globdat );

      w.parser->setGroups ( w.groups );

      k = w.makeGroups ( myName_,
                         myConf .makeProps( propName ),
                         myProps.getProps ( propName ) );

      if ( k == 0 )
      {
        print ( out, "Created zero new ",
                w.itemName, " groups\n\n" );
      }
      else if ( k == 1 )
      {
        print ( out, "Created one new ", w.itemName, " group\n\n" );
      }
      else
      {
        print ( out, "Created ", k, " new ", w.itemName,
                " groups\n\n" );
      }
    }
  }

  return DONE;
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Module> GroupgenModule::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  return newInstance<Self> ( name );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void GroupgenModule::declare ()
{
  ModuleFactory::declare ( TYPE_NAME,  & makeNew );
  ModuleFactory::declare ( CLASS_NAME, & makeNew );
}


JIVE_END_PACKAGE( app )
