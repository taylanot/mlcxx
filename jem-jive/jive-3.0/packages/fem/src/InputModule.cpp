
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
#include <jem/io/FileName.h>
#include <jem/io/IOException.h>
#include <jem/mp/Context.h>
#include <jive/util/utilities.h>
#include <jive/util/Assignable.h>
#include <jive/util/ParserActions.h>
#include <jive/mp/Globdat.h>
#include <jive/app/ModuleFactory.h>
#include <jive/fem/typedefs.h>
#include <jive/fem/Names.h>
#include <jive/fem/Globdat.h>
#include <jive/fem/DataParser.h>
#include <jive/fem/XNodeSet.h>
#include <jive/fem/XElementSet.h>
#include <jive/fem/XBoundarySet.h>
#include <jive/fem/InputModule.h>


JEM_DEFINE_CLASS( jive::fem::InputModule );


JIVE_BEGIN_PACKAGE( fem )


using jem::newInstance;
using jem::System;
using jem::io::Writer;
using jem::io::IOException;
using jive::util::StorageMode;


//=======================================================================
//   class InputModule
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  InputModule::TYPE_NAME    = "FEInput";

const int    InputModule::READ_NODES   = 1 << 0;
const int    InputModule::READ_ELEMS   = 1 << 1;

const int    InputModule::READ_MESH    = READ_NODES | READ_ELEMS;

const int    InputModule::READ_BOUNDS  = 1 << 2;
const int    InputModule::READ_CONS    = 1 << 3;
const int    InputModule::READ_GROUPS  = 1 << 4;
const int    InputModule::READ_TABLES  = 1 << 5;
const int    InputModule::READ_DBASES  = 1 << 6;
const int    InputModule::READ_FUNCS   = 1 << 7;

const int    InputModule::READ_ALL     = ~0x0;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


InputModule::InputModule

  ( const String&  name,
    int            mask ) :

    Super ( name )

{
  readMask_ = mask;
}


InputModule::~InputModule ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


Module::Status InputModule::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  using jem::io::FileName;
  using jive::util::getStorageMode;
  using jive::util::ParserActions;
  using jive::util::ParserOptions;

  Writer&           info = jem::System::info ( myName_ );

  Ref<MPContext>    mpx  = mp::Globdat::getMPContext ( globdat );

  const int         mpSize  = mpx->size ();

  Properties        myConf  = conf .makeProps ( myName_ );
  Properties        myProps = props.findProps ( myName_ );

  StorageMode       smode   = util::DEFAULT_STORAGE;

  Ref<DataParser>   parser;

  String            fname;
  String            baseName;

  idx_t             maxParts;
  int               mask;


  getStorageMode ( smode,   conf,   props );
  getStorageMode ( smode, myConf, myProps );

  if ( props.find( fname, CASE_NAME ) )
  {
    fname = fname + ".in";
  }

  maxParts = 0;

  myProps.find ( fname,    PropNames::FILE );
  myProps.find ( maxParts, PropNames::MAX_PARTS, 0, 10000 );

  myConf.set ( PropNames::FILE,      fname    );
  myConf.set ( PropNames::MAX_PARTS, maxParts );

  fname = util::expandString ( fname, globdat );
  mask  = readMask_;

  if ( mpSize > 1 )
  {
    mask = mask & (~READ_MESH);
  }

  baseName = FileName::getBaseFileName ( fname );

  initData_ ( globdat, baseName, mask, smode );

  if ( ! fname.size() )
  {
    print ( info, myName_, " : no data input file specified\n\n" );

    return DONE;
  }

  parser = newInstance<DataParser> ( mpx );

  initParser_ ( *parser, myConf, myProps, globdat, mask );

  parser->setMaxParts     ( maxParts );
  parser->skipUnknownTags ( true     );

  if ( mpSize > 1 )
  {
    Properties  params ( "parserParams" );

    params.set         ( ParserOptions::SKIP_UNKNOWN, true   );
    parser->takeAction ( ParserActions::SET_OPTIONS,  params );
  }

  print ( System::out(),
          "Reading data from `", fname, "\' ...\n\n" );

  try
  {
    parser->parseFile ( fname, &info );
  }
  catch ( IOException& ex )
  {
    ex.setContext ( getContext() );
    throw;
  }

  print ( info, '\n' );

  return DONE;
}


//-----------------------------------------------------------------------
//   initData
//-----------------------------------------------------------------------


void InputModule::initData

  ( const Properties&  globdat,
    const String&      baseName,
    int                mask,
    StorageMode        smode )

{
  using jive::util::joinNames;
  using jive::util::Assignable;

  Assignable<NodeSet>      nodes;
  Assignable<ElementSet>   elems;
  Assignable<BoundarySet>  bounds;


  if ( mask & READ_NODES )
  {
    nodes = NodeSet::find ( globdat );

    if ( ! nodes )
    {
      nodes = newXNodeSet ( joinNames ( baseName, Globdat::NODES ),
                            smode );

      nodes.store ( globdat );
    }
  }

  if ( (mask & READ_ELEMS) && nodes )
  {
    elems = ElementSet::find ( globdat );

    if ( ! elems )
    {
      elems = newXElementSet (
        joinNames ( baseName, Globdat::ELEMENTS ),
        nodes,
        smode
      );

      elems.store ( globdat );
    }
  }

  if ( (mask & READ_BOUNDS) && elems )
  {
    bounds = BoundarySet::find ( globdat );

    if ( ! bounds )
    {
      bounds = newXBoundarySet (
        joinNames ( baseName, Globdat::BOUNDARIES ),
        elems,
        smode
      );

      bounds.store ( globdat );
    }
  }
}


//-----------------------------------------------------------------------
//   initParser
//-----------------------------------------------------------------------


void InputModule::initParser

  ( DataParser&        parser,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat,
    int                mask )

{
  XNodeSet      nodes  = XNodeSet     :: find ( globdat );
  XElementSet   elems  = XElementSet  :: find ( globdat );
  XBoundarySet  bounds = XBoundarySet :: find ( globdat );


  if ( (mask & READ_NODES) && nodes )
  {
    parser.addNodesParser ( nodes );
  }

  if ( (mask & READ_ELEMS) && elems )
  {
    parser.addElementsParser ( elems );
  }

  if ( (mask & READ_BOUNDS) && bounds )
  {
    parser.addBoundsParser ( bounds );
  }

  if ( mask & READ_CONS )
  {
    parser.addConstraintsParsers ( globdat );
  }

  if ( mask & READ_GROUPS )
  {
    parser.addItemGroupsParsers ( globdat );
  }

  if ( mask & READ_TABLES )
  {
    String  filter = "*";

    props.find ( filter, PropNames::TABLE_FILTER );
    conf .set  ( PropNames::TABLE_FILTER, filter );

    parser.addTablesParsers ( globdat, filter );
  }

  if ( mask & READ_DBASES )
  {
    String  filter = "*";

    props.find ( filter, PropNames::DBASE_FILTER );
    conf .set  ( PropNames::DBASE_FILTER, filter );

    parser.addDbasesParsers ( globdat, filter );
  }

  if ( mask & READ_FUNCS )
  {
    String  filter = "*";

    props.find ( filter, PropNames::FUNC_FILTER );
    conf .set  ( PropNames::FUNC_FILTER, filter );

    parser.addFuncsParser ( globdat, filter );
  }
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Module> InputModule::makeNew

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


void InputModule::declare ()
{
  using jive::app::ModuleFactory;

  ModuleFactory::declare ( TYPE_NAME,  & makeNew );
  ModuleFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   initData_
//-----------------------------------------------------------------------


void InputModule::initData_

  ( const Properties&  globdat,
    const String&      baseName,
    int                mask,
    StorageMode        smode )

{
  initData ( globdat, baseName, mask, smode );
}


//-----------------------------------------------------------------------
//   initParser_
//-----------------------------------------------------------------------


void InputModule::initParser_

  ( DataParser&        parser,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat,
    int                mask )

{
  initParser ( parser, conf, props, globdat, mask );
}


JIVE_END_PACKAGE( fem )
