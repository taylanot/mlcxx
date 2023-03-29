
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
#include <jem/io/list.h>
#include <jem/io/NullWriter.h>
#include <jem/io/ArrayInputStream.h>
#include <jem/io/ArrayOutputStream.h>
#include <jem/io/DataInputStream.h>
#include <jem/io/DataOutputStream.h>
#include <jem/mp/Buffer.h>
#include <jem/mp/Context.h>
#include <jem/util/Pattern.h>
#include <jem/util/Properties.h>
#include <jem/util/Dictionary.h>
#include <jem/util/StringUtils.h>
#include <jive/util/utilities.h>
#include <jive/util/Globdat.h>
#include <jive/util/Runtime.h>
#include <jive/util/ItemSet.h>
#include <jive/util/ItemGroup.h>
#include <jive/util/Table.h>
#include <jive/util/Database.h>
#include <jive/util/DofSpace.h>
#include <jive/util/Constraints.h>
#include <jive/util/FuncUtils.h>
#include <jive/mp/Globdat.h>
#include <jive/app/import.h>
#include <jive/app/typedefs.h>
#include <jive/app/Actions.h>
#include <jive/app/Commands.h>
#include <jive/app/ModuleFactory.h>
#include <jive/app/InfoModule.h>


JEM_DEFINE_CLASS( jive::app::InfoModule );


JIVE_BEGIN_PACKAGE( app )


using jem::newInstance;
using jem::System;
using jem::io::endl;
using jem::io::flush;
using jem::io::Writer;
using jem::mp::RecvBuffer;
using jem::mp::SendBuffer;
using jem::util::Dict;
using jem::util::DictEnum;
using jive::util::ItemSet;
using jive::util::ItemGroup;
using jive::util::Database;


//=======================================================================
//   class InfoModule::Utils_
//=======================================================================


class InfoModule::Utils_
{
 public:

  static StringVector           listItemSets

    ( MPContext&                  mpx,
      const Properties&           globdat );

  static void                   getTotals

    ( const IdxVector&            totals,
      const StringVector&         isetNames,
      const Properties&           globdat );

  static void                   getTotals

    ( MPContext&                  mpx,
      const IdxVector&            totals,
      const StringVector&         isetNames,
      const Properties&           globdat );

  static void                   printSummary

    ( PrintWriter&                out,
      const IdxVector&            totals,
      const StringVector&         isetNames,
      const Properties&           globdat );

};


//-----------------------------------------------------------------------
//   listItemSets
//-----------------------------------------------------------------------


StringVector InfoModule::Utils_::listItemSets

  ( MPContext&         mpx,
    const Properties&  globdat )

{
  using jem::byte;
  using jem::Array;
  using jem::io::ArrayInputStream;
  using jem::io::ArrayOutputStream;
  using jem::io::DataInputStream;
  using jem::io::DataOutputStream;

  StringVector  names;
  Array<byte>   buf;
  idx_t         n;


  if ( mpx.myRank() == 0 )
  {
    Ref<ArrayOutputStream>  bufStream;
    Ref<DataOutputStream>   datStream;

    bufStream = newInstance<ArrayOutputStream> ();
    datStream = newInstance<DataOutputStream>  ( bufStream );

    names.ref ( ItemSet::listAll( globdat ) );

    encode ( *datStream, names );

    datStream->close ();

    buf.ref ( bufStream->toArray() );

    if ( ! buf.isContiguous() )
    {
      buf.ref ( buf.clone() );
    }

    n = buf.size ();

    mpx.broadcast ( SendBuffer( &n, 1 ) );
    mpx.broadcast ( SendBuffer( buf.addr(), n ) );
  }
  else
  {
    Ref<ArrayInputStream>  bufStream;
    Ref<DataInputStream>   datStream;

    mpx.broadcast ( RecvBuffer( &n, 1 ), 0 );
    buf.resize    ( n );
    mpx.broadcast ( RecvBuffer( buf.addr(), n ), 0 );

    bufStream = newInstance<ArrayInputStream> ( buf );
    datStream = newInstance<DataInputStream>  ( bufStream );

    decode ( *datStream, names );
  }

  return names;
}


//-----------------------------------------------------------------------
//   getTotals (single-process case)
//-----------------------------------------------------------------------


void InfoModule::Utils_::getTotals

  ( const IdxVector&     totals,
    const StringVector&  isetNames,
    const Properties&    globdat )

{
  using jive::util::Constraints;

  JEM_PRECHECK ( totals.size() == 3 * isetNames.size() );

  const idx_t       isetCount = isetNames.size ();

  Ref<DofSpace>     dofs;
  Ref<Constraints>  cons;
  Ref<ItemSet>      items;
  idx_t             iset;
  idx_t             i;


  totals = 0;

  for ( iset = 0; iset < isetCount; iset++ )
  {
    items = ItemSet::find ( isetNames[iset], globdat );

    if ( ! items )
    {
      continue;
    }

    i = 3 * iset;

    totals[i + 0] = items->size ();

    dofs = DofSpace::find ( items, globdat );

    if ( ! dofs )
    {
      continue;
    }

    totals[i + 1] = dofs->dofCount ();

    cons = Constraints::find ( dofs, globdat );

    if ( cons )
    {
      totals[i + 2] = cons->slaveDofCount ();
    }
  }
}


//-----------------------------------------------------------------------
//   getTotals (multi-process case)
//-----------------------------------------------------------------------


void InfoModule::Utils_::getTotals

  ( MPContext&           mpx,
    const IdxVector&     totals,
    const StringVector&  isetNames,
    const Properties&    globdat )

{
  const idx_t  n = totals.size ();

  IdxVector    buf ( n );


  getTotals     ( buf, isetNames, globdat );

  mpx.allreduce ( RecvBuffer( totals.addr(), n ),
                  SendBuffer( buf   .addr(), n ),
                  jem::mp::SUM );
}


//-----------------------------------------------------------------------
//   printSummary
//-----------------------------------------------------------------------


void InfoModule::Utils_::printSummary

  ( PrintWriter&         out,
    const IdxVector&     totals,
    const StringVector&  isetNames,
    const Properties&    globdat )

{

  using jem::io::dots;
  using jive::util::FuncUtils;

  const idx_t   isetCount = isetNames.size ();

  Ref<Dict>     dict;
  Ref<ItemSet>  items;
  String        itemName;
  idx_t         itemCount;
  idx_t         dofCount;
  idx_t         conCount;
  idx_t         iset;
  idx_t         i;


  for ( iset = 0; iset < isetCount; iset++ )
  {
    items = ItemSet::find ( isetNames[iset], globdat );

    if ( ! items )
    {
      continue;
    }

    i = 3 * iset;

    itemName  = items->getItemName  ();
    itemCount = totals[i + 0];
    dofCount  = totals[i + 1];
    conCount  = totals[i + 2];

    print ( out, endl, items->getItemsName(), dots, itemCount );

    if ( dofCount > 0 )
    {
      print ( out, endl, itemName, " dofs", dots, dofCount );
    }

    if ( conCount > 0 )
    {
      print ( out, endl, itemName, " constraints", dots, conCount );
    }

    dict = ItemGroup::findFor ( items, globdat );

    if ( dict )
    {
      print ( out, endl, itemName, " groups", dots, dict->size() );
    }

    dict = Table::findFor ( items, globdat );

    if ( dict )
    {
      print ( out, endl, itemName, " tables", dots, dict->size() );
    }

    dict = Database::findFor ( items, globdat );

    if ( dict )
    {
      print ( out, endl, itemName, " data bases",
              dots, dict->size() );
    }
  }

  dict = FuncUtils::findFuncs ( globdat );

  if ( dict )
  {
    print ( out, endl, "functions", dots, dict->size() );
  }

  print ( out, endl );
}


//=======================================================================
//   class InfoModule
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  InfoModule::TYPE_NAME = "Info";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


InfoModule::InfoModule ( const String& name ) :

  Super ( name )

{
  tlast_ = 0.0;
}


InfoModule::~InfoModule ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


app::Module::Status InfoModule::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  using jem::io::indent;
  using jem::io::outdent;

  Ref<PrintWriter>  out =

    newInstance<PrintWriter> ( & System::out() );

  print        ( *out, "Data summary:", indent, endl );
  printSummary ( *out, globdat );
  print        ( *out, outdent, endl, flush );

  tlast_ = 0.0;

  return OK;
}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


Module::Status InfoModule::run ( const Properties& globdat )
{
  using jive::util::Runtime;

  Writer*  out;
  double   rt, st;
  idx_t    istep;


  rt = Runtime::getValue ( globdat );

  if ( rt - tlast_ > 2.0 )
  {
    out    = & System::out ();
    tlast_ = rt;
  }
  else
  {
    out    = & System::info ( myName_ );
  }

  istep = 0;

  globdat.find ( istep, util::Globdat::TIME_STEP );

  if ( globdat.find( st, util::Globdat::TIME ) )
  {
    print ( *out,   "Time step       : ", istep );
    print ( *out, "\nSimulation time : ", st );

  }
  else
  {
    print ( *out,   "Iteration       : ", istep );
  }

  print ( *out, "\nElapsed runtime : ", rt, " seconds\n\n" );

  return OK;
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool InfoModule::takeAction

  ( const String&      action,
    const Properties&  params,
    const Properties&  globdat )

{
  bool  result = false;

  if      ( action == Actions::EXE_COMMAND )
  {
    String  cmd, arg;

    params.get ( cmd, ActionParams::COMMAND );
    params.get ( arg, ActionParams::COMMAND_ARG );

    if ( cmd == Commands::LIST )
    {
      exeListCmd_ ( arg, globdat );

      result = true;
    }
  }
  else if ( action == Actions::GET_COMMANDS )
  {
    params.set ( Commands::LIST, "item-sets groups tables dbases" );

    result = true;
  }
  else if ( action == Actions::PRINT_HELP )
  {
    Ref<PrintWriter>  out;

    params.get ( out, ActionParams::PRINTER );

    printHelp ( *out );

    result = true;
  }

  return result;
}


//-----------------------------------------------------------------------
//   printHelp
//-----------------------------------------------------------------------


void InfoModule::printHelp ( PrintWriter& out ) const
{
  using jem::io::beginItem;
  using jem::io::endItem;

  print ( out, beginItem( "list <what> [pattern]" ) );
  print ( out,   "List the names of the objects in the " );
  print ( out,   "category <what>. Possible categories are " );
  print ( out,   "`item-sets\', `groups\', `tables\' and " );
  print ( out,   "`dbases\'. If the optional argument <pattern> " );
  print ( out,   "is specified, then only the names matching " );
  print ( out,   "<pattern> are printed. Else, all names are " );
  print ( out,   "printed." );
  print ( out, endItem );
}


//-----------------------------------------------------------------------
//   printSummary
//-----------------------------------------------------------------------


void InfoModule::printSummary

  ( PrintWriter&       out,
    const Properties&  globdat )

{
  Ref<MPContext>  mpx = mp::Globdat::getMPContext ( globdat );

  StringVector    isetNames;
  IdxVector       totals;


  if ( mpx->size() == 1 )
  {
    isetNames.ref     ( ItemSet::listAll( globdat ) );
    totals.resize     ( 3 * isetNames.size() );
    Utils_::getTotals ( totals, isetNames, globdat );
  }
  else
  {
    isetNames.ref     ( Utils_::listItemSets( *mpx, globdat ) );
    totals.resize     ( 3 * isetNames.size() );
    Utils_::getTotals ( *mpx, totals, isetNames, globdat );
  }

  Utils_::printSummary ( out, totals, isetNames, globdat );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Module> InfoModule::makeNew

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


void InfoModule::declare ()
{
  ModuleFactory::declare ( TYPE_NAME,  & makeNew );
  ModuleFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   exeListCmd_
//-----------------------------------------------------------------------


void InfoModule::exeListCmd_

  ( const String&      arg,
    const Properties&  globdat )

{
  using jem::IllegalInputException;
  using jem::io::endl;
  using jem::io::indent;
  using jem::io::NullWriter;
  using jem::util::Pattern;
  using jem::util::StringUtils;

  Ref<Dict>            (*findForFunc)

    ( const Ref<ItemSet>&  items,
      const Properties&    globdat );

  StringVector      list = StringUtils::split ( arg );

  Ref<PrintWriter>  pr;
  Ref<ItemSet>      items;
  Ref<Dict>         dict;
  Ref<DictEnum>     e;

  String            what;
  String            pattern;


  if ( list.size() < 1 )
  {
    throw IllegalInputException ( getContext(), "missing category" );
  }

  if ( list.size() > 2 )
  {
    throw IllegalInputException (
      getContext (),
      "too many arguments: " + arg
    );
  }

  what = list[0];

  if ( list.size() == 2 )
  {
    pattern = list[1];

    Pattern::check ( pattern );
  }

  findForFunc = 0;

  if      ( what == "groups" )
  {
    findForFunc = ItemGroup::findFor;
  }
  else if ( what == "tables" )
  {
    findForFunc = Table::findFor;
  }
  else if ( what == "dbases" )
  {
    findForFunc = Database::findFor;
  }
  else if ( what != "item-sets" )
  {
    throw IllegalInputException (
      getContext (),
      "invalid category: " + what
    );
  }

  list.ref ( ItemSet::listAll( globdat ) );

  if ( mp::Globdat::myRank( globdat ) == 0 )
  {
    pr = newInstance<PrintWriter> ( & System::out() );
  }
  else
  {
    pr = newInstance<PrintWriter> ( newInstance<NullWriter>() );
  }

  print ( *pr, endl, indent );

  if ( findForFunc )
  {
    for ( idx_t i = 0; i < list.size(); i++ )
    {
      items = ItemSet::find ( list[i], globdat );

      if ( ! items )
      {
        continue;
      }

      dict = findForFunc ( items, globdat );

      if ( ! dict )
      {
        continue;
      }

      for ( e = dict->enumerate(); ! e->atEnd(); e->toNext() )
      {
        if ( pattern.size() )
        {
          String  name = list[i] + '/' + e->getKey ();

          if ( Pattern::matches( pattern, name ) )
          {
            print ( *pr, name );
          }
        }
        else
        {
          print ( *pr, list[i], '/', e->getKey() );
        }

        print ( *pr, endl );
      }
    }
  }
  else
  {
    for ( idx_t i = 0; i < list.size(); i++ )
    {
      if ( pattern.size() == 0 ||
           Pattern::matches( pattern, list[i] ) )
      {
        print ( *pr, list[i], endl );
      }
    }
  }

  print ( *pr, endl, flush );
}


JIVE_END_PACKAGE( app )
