
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
#include <jem/io/GzipInputStream.h>
#include <jem/io/InputStreamReader.h>
#include <jem/mp/UniContext.h>
#include <jem/mp/BcastStream.h>
#include <jem/util/Properties.h>
#include <jem/util/Dictionary.h>
#include <jem/util/DictionaryEnumerator.h>
#include <jem/xml/TextLog.h>
#include <jem/xml/Tokenizer.h>
#include <jem/xml/SkipTagParser.h>
#include <jem/xml/SimpleTagParser.h>
#include <jive/log.h>
#include <jive/util/utilities.h>
#include <jive/util/ItemGroup.h>
#include <jive/util/XPointSet.h>
#include <jive/util/XGroupSet.h>
#include <jive/util/XMemberSet.h>
#include <jive/util/PointSetParser.h>
#include <jive/util/GroupSetParser.h>
#include <jive/util/MemberSetParser.h>
#include <jive/util/ItemGroupsParser.h>
#include <jive/util/XTable.h>
#include <jive/util/TableParser.h>
#include <jive/util/TablesParser.h>
#include <jive/util/Database.h>
#include <jive/util/DBaseParser.h>
#include <jive/util/DBasesParser.h>
#include <jive/util/PartParser.h>
#include <jive/util/IncludeParser.h>
#include <jive/util/ConstraintsParser.h>
#include <jive/util/FuncUtils.h>
#include <jive/util/FuncsParser.h>
#include <jive/util/DataParser.h>


JEM_DEFINE_CLASS( jive::util::DataParser );


JIVE_BEGIN_PACKAGE( util )


using jem::staticCast;
using jem::newInstance;
using jem::io::InputStream;
using jem::io::GzipInputStream;
using jem::io::InputStreamReader;
using jem::mp::UniContext;
using jem::mp::BcastStream;
using jem::xml::TextLog;
using jem::xml::ParseLog;
using jem::xml::Tokenizer;
using jem::xml::SkipTagParser;
using jem::xml::SimpleTagParser;


//=======================================================================
//   class DataParser::Utils_
//=======================================================================


class DataParser::Utils_
{
 public:

  static inline String    getTagName

    ( const Ref<ItemSet>&   items,
      const String&         tag )

  {
    if ( tag.size() == 0 && items )
    {
      return items->getTagName ();
    }
    else
    {
      return tag;
    }
  }

};


//=======================================================================
//   class DataParser
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


DataParser::DataParser ( const Ref<MPContext>& mpx ) :

  mpx_ ( mpx )

{
  if ( ! mpx_ )
  {
    mpx_ = newInstance<UniContext> ();
  }

  addParser ( IncludeParser::TAG_NAME,
              newInstance<IncludeParser>( mpx_ ) );
}


DataParser::~DataParser ()
{}


//-----------------------------------------------------------------------
//   parseFile
//-----------------------------------------------------------------------


void DataParser::parseFile

  ( const String&       fname,
    const Ref<Writer>&  output )

{
  Ref<InputStream>  in = BcastStream::open ( 0, mpx_, fname );

  Ref<Tokenizer>    tokzer;
  Ref<ParseLog>     logger;


  if ( isGzipFile( fname ) )
  {
    in = newInstance<GzipInputStream> ( in );
  }

  if ( output )
  {
    logger = newInstance<TextLog> ( output );
  }

  tokzer = newInstance<Tokenizer> (
    newInstance<InputStreamReader> ( in )
  );

  Super::parseInput ( tokzer, fname, logger );
}


//-----------------------------------------------------------------------
//   setMaxParts
//-----------------------------------------------------------------------


void DataParser::setMaxParts ( idx_t maxParts )
{
  if ( maxParts <= 0 )
  {
    addParser ( PartParser::TAG_NAME, nullptr );
  }
  else
  {
    addParser ( PartParser::TAG_NAME,
                newInstance<PartParser>( maxParts ) );
  }
}


//-----------------------------------------------------------------------
//   skipUnknownTags
//-----------------------------------------------------------------------


void DataParser::skipUnknownTags ( bool choice )
{
  if ( choice )
  {
    setDefaultParser ( newInstance<SkipTagParser>() );
  }
  else
  {
    setDefaultParser ( nullptr );
  }
}


//-----------------------------------------------------------------------
//   addPointSetParser
//-----------------------------------------------------------------------


void DataParser::addPointSetParser

  ( const Ref<XPointSet>&  points,
    idx_t                  rank,
    const String&          tag )

{
  addParser (
    Utils_::getTagName ( points, tag ),
    newInstance<SimpleTagParser> (
      newInstance<PointSetParser> ( points, rank )
    )
  );
}


//-----------------------------------------------------------------------
//   addGroupSetParser
//-----------------------------------------------------------------------


void DataParser::addGroupSetParser

  ( const Ref<XGroupSet>& groups,
    idx_t                 groupSize,
    const String&         tag )

{
  addParser (
    Utils_::getTagName ( groups, tag ),
    newInstance<SimpleTagParser> (
      newInstance<GroupSetParser> ( groups, groupSize )
    )
  );
}


//-----------------------------------------------------------------------
//   addMemberSetParser
//-----------------------------------------------------------------------


void DataParser::addMemberSetParser

  ( const Ref<XMemberSet>&  members,
    idx_t                   maxLocal,
    const String&           tag )

{
  addParser (
    Utils_::getTagName ( members, tag ),
    newInstance<SimpleTagParser> (
      newInstance<MemberSetParser> ( members, maxLocal )
    )
  );
}


//-----------------------------------------------------------------------
//   addItemGroupsParser
//-----------------------------------------------------------------------


void DataParser::addItemGroupsParser

  ( const Ref<ItemSet>&  items,
    const Ref<Dict>&     groups,
    const String&        filter,
    const String&        tag )

{
  String  tagName;

  if ( tag.size() == 0 && items )
  {
    tagName = ItemGroup::getTagName ( items->getItemName() );
  }
  else
  {
    tagName = tag;
  }

  addParser (
    tagName,
    newInstance<ItemGroupsParser> ( items, groups, filter )
  );
}


//-----------------------------------------------------------------------
//   addItemGroupsParsers
//-----------------------------------------------------------------------


void DataParser::addItemGroupsParsers

  ( const Properties&  globdat,
    const String&      filter )

{
  Ref<Dict>      list = ItemSet::getAll ( globdat );
  Ref<ItemSet>   items;
  Ref<DictEnum>  e;

  for ( e = list->enumerate(); ! e->atEnd(); e->toNext() )
  {
    items = staticCast<ItemSet> ( e->getValue() );

    addItemGroupsParser (
      items,
      ItemGroup::getFor ( items, globdat ),
      filter
    );
  }
}


//-----------------------------------------------------------------------
//   addTableParser
//-----------------------------------------------------------------------


void DataParser::addTableParser

  ( const Ref<XTable>&  table,
    const Ref<Dict>&    rowGroups,
    const String&       tag )

{
  String  tagName;

  if ( tag.size() == 0 && table )
  {
    tagName = table->getTagName ();
  }
  else
  {
    tagName = tag;
  }

  addParser (
    tagName,
    newInstance<SimpleTagParser> (
      newInstance<TableParser> ( mpx_, table, rowGroups )
    )
  );
}


//-----------------------------------------------------------------------
//   addTablesParser
//-----------------------------------------------------------------------


void DataParser::addTablesParser

  ( const Ref<ItemSet>&  items,
    const Ref<Dict>&     tables,
    const Ref<Dict>&     rowGroups,
    const String&        filter,
    const String&        tag )

{
  String  tagName;

  if ( tag.size() == 0 && items )
  {
    tagName = Table::getTagName ( items->getItemName() );
  }
  else
  {
    tagName = tag;
  }

  addParser (
    tagName,
    newInstance<TablesParser> ( mpx_,   items,
                                tables, rowGroups, filter )
  );
}


//-----------------------------------------------------------------------
//   addTablesParsers
//-----------------------------------------------------------------------


void DataParser::addTablesParsers

  ( const Properties&  globdat,
    const String&      filter )

{
  Ref<Dict>      list = ItemSet::getAll ( globdat );
  Ref<ItemSet>   items;
  Ref<DictEnum>  e;

  for ( e = list->enumerate(); ! e->atEnd(); e->toNext() )
  {
    items = staticCast<ItemSet> ( e->getValue() );

    addTablesParser (
      items,
      Table    ::getFor  ( items, globdat ),
      ItemGroup::findFor ( items, globdat ),
      filter
    );
  }
}


//-----------------------------------------------------------------------
//   addDbaseParser
//-----------------------------------------------------------------------


void DataParser::addDbaseParser

  ( const Ref<Database>&  dbase,
    const Ref<Dict>&      rowGroups,
    const String&         tag )

{
  String  tagName;

  if ( tag.size() == 0 && dbase )
  {
    tagName = dbase->getTagName ();
  }
  else
  {
    tagName = tag;
  }

  addParser (
    tagName,
    newInstance<SimpleTagParser> (
      newInstance<DBaseParser> ( mpx_, dbase, rowGroups )
    )
  );
}


//-----------------------------------------------------------------------
//   addDbasesParser
//-----------------------------------------------------------------------


void DataParser::addDbasesParser

  ( const Ref<ItemSet>&  items,
    const Ref<Dict>&     dbases,
    const Ref<Dict>&     rowGroups,
    const String&        filter,
    const String&        tag )

{
  String  tagName;

  if ( tag.size() == 0 && items )
  {
    tagName = Database::getTagName ( items->getItemName() );
  }
  else
  {
    tagName = tag;
  }

  addParser (
    tagName,
    newInstance<DBasesParser> ( mpx_,   items,
                                dbases, rowGroups, filter )
  );
}


//-----------------------------------------------------------------------
//   addDbasesParsers
//-----------------------------------------------------------------------


void DataParser::addDbasesParsers

  ( const Properties&  globdat,
    const String&      filter )

{
  Ref<Dict>      list = ItemSet::getAll ( globdat );
  Ref<ItemSet>   items;
  Ref<DictEnum>  e;

  for ( e = list->enumerate(); ! e->atEnd(); e->toNext() )
  {
    items = staticCast<ItemSet> ( e->getValue() );

    addDbasesParser (
      items,
      Database ::getFor  ( items, globdat ),
      ItemGroup::findFor ( items, globdat ),
      filter
    );
  }
}


//-----------------------------------------------------------------------
//   addConstraintsParser
//-----------------------------------------------------------------------


void DataParser::addConstraintsParser

  ( const Ref<ConParser>&  conParser,
    const String&          tag )

{
  String  tagName = tag;

  if ( tag.size() == 0 )
  {
    tagName = conParser->getTagName ();
  }

  addParser (
    tagName,
    newInstance<SimpleTagParser> ( conParser )
  );
}


//-----------------------------------------------------------------------
//   addConstraintsParsers
//-----------------------------------------------------------------------


void DataParser::addConstraintsParsers ( const Properties& globdat )
{
  Ref<Dict>       list = ItemSet::getAll ( globdat );
  Ref<ItemSet>    items;
  Ref<ConParser>  conParser;
  Ref<DictEnum>   e;

  for ( e = list->enumerate(); ! e->atEnd(); e->toNext() )
  {
    items = staticCast<ItemSet> ( e->getValue() );

    conParser = newInstance<ConParser> (
      items,
      ItemGroup::getFor ( items, globdat )
    );

    addParser (
      conParser->getTagName (),
      newInstance<SimpleTagParser> ( conParser )
    );

    conParser->store ( globdat );
  }
}


//-----------------------------------------------------------------------
//   addFuncsParser
//-----------------------------------------------------------------------


void DataParser::addFuncsParser

  ( const Ref<Dictionary>&  funcs,
    const String&           filter,
    const String&           tag )

{
  String  tagName = tag;

  if ( ! tag.size() )
  {
    tagName = "Function";
  }

  addParser ( tagName, newInstance<FuncsParser>( funcs, filter ) );
}


void DataParser::addFuncsParser

  ( const Properties&  globdat,
    const String&      filter,
    const String&      tag )

{
  addFuncsParser ( FuncUtils::getFuncs( globdat ), filter, tag );
}


JIVE_END_PACKAGE( util )
