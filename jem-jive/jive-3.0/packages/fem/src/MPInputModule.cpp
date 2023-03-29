
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
#include <jem/util/HashDictionary.h>
#include <jem/util/DictionaryEnumerator.h>
#include <jive/util/utilities.h>
#include <jive/util/ItemGroup.h>
#include <jive/util/StorageMode.h>
#include <jive/util/Assignable.h>
#include <jive/util/ConstraintsParser.h>
#include <jive/app/ModuleFactory.h>
#include <jive/mp/Globdat.h>
#include <jive/fem/typedefs.h>
#include <jive/fem/Names.h>
#include <jive/fem/Globdat.h>
#include <jive/fem/DataParser.h>
#include <jive/fem/XNodeSet.h>
#include <jive/fem/XElementSet.h>
#include <jive/fem/MPInputModule.h>


JEM_DEFINE_CLASS( jive::fem::MPInputModule );


JIVE_BEGIN_PACKAGE( fem )


using jem::newInstance;
using jem::io::Writer;
using jem::util::Dict;


//=======================================================================
//   class MPInputModule::Utils_
//=======================================================================


class MPInputModule::Utils_
{
 public:

  static Ref<Dict>        getGroupsFor

    ( const Ref<ItemSet>&   items,
      const Properties&     globdat );

};


//-----------------------------------------------------------------------
//   getGroupsFor
//-----------------------------------------------------------------------


Ref<Dict> MPInputModule::Utils_::getGroupsFor

  ( const Ref<ItemSet>&  items,
    const Properties&    globdat )

{
  using jem::util::HashDict;
  using jem::util::DictEnum;

  Ref<Dict>      dict   = ItemGroup::getFor     ( items, globdat );
  Ref<Dict>      groups = newInstance<HashDict> ();
  Ref<DictEnum>  e      = dict->enumerate       ();

  for ( ; ! e->atEnd(); e->toNext() )
  {
    groups->insert ( e->getKey(), e->getValue() );
  }

  return groups;
}


//=======================================================================
//   class MPInputModule
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  MPInputModule::TYPE_NAME = "FEMPInput";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


MPInputModule::MPInputModule ( const String& name ) :

  Super ( name )

{}


MPInputModule::~MPInputModule ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


Module::Status MPInputModule::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  Ref<MPContext> mpx = mp::Globdat::getMPContext ( globdat );

  if ( mpx->size() > 1 && mpx->myRank() == 0 )
  {
    readMesh_ ( conf, props, globdat );
  }

  return DONE;
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Module> MPInputModule::makeNew

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


void MPInputModule::declare ()
{
  using jive::app::ModuleFactory;

  ModuleFactory::declare ( TYPE_NAME,  & makeNew );
  ModuleFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   readMesh_
//-----------------------------------------------------------------------


void MPInputModule::readMesh_

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  using jem::System;
  using jem::io::FileName;
  using jem::io::IOException;
  using jive::util::joinNames;
  using jive::util::getStorageMode;
  using jive::util::ConParser;
  using jive::util::StorageMode;
  using jive::util::Assignable;

  Assignable
    <XNodeSet>     nodes;
  Assignable
    <XElementSet>  elems;

  Ref<ConParser>   nodeConParser;
  Ref<ConParser>   elemConParser;

  Ref<Dict>        nodeGroups;
  Ref<Dict>        elemGroups;

  Writer&          info    = System::info    ( myName_ );

  Properties       myConf  = conf .makeProps ( myName_ );
  Properties       myProps = props.findProps ( myName_ );

  StorageMode      smode   = util::DEFAULT_STORAGE;

  Ref<DataParser>  parser;
  String           fname;
  String           baseName;

  idx_t            maxParts;


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

  fname    = util::expandString ( fname, globdat );
  baseName = FileName::getBaseFileName ( fname );

  nodes    = XNodeSet    :: find ( globdat );
  elems    = XElementSet :: find ( globdat );

  if ( ! nodes )
  {
    nodes = newXNodeSet ( joinNames ( baseName, Globdat::NODES ),
                          smode );

    nodes.store ( globdat );
  }

  if ( ! elems )
  {
    elems = newXElementSet (
      joinNames ( baseName, Globdat::ELEMENTS ),
      nodes,
      smode
    );

    elems.store ( globdat );
  }

  if ( ! fname.size() )
  {
    print ( info, myName_, " : no mesh input file specified\n\n" );

    return;
  }

  nodeGroups    = Utils_::getGroupsFor   ( nodes.getData(),
                                           globdat );

  elemGroups    = Utils_::getGroupsFor   ( elems.getData(),
                                           globdat );

  nodeConParser = newInstance<ConParser> ( nodes.getData (),
                                           nodeGroups );

  elemConParser = newInstance<ConParser> ( elems.getData (),
                                           elemGroups );

  parser = newInstance<DataParser> ();

  parser->addNodesParser       ( nodes );
  parser->addElementsParser    ( elems );
  parser->addItemGroupsParser  ( nodes.getData(), nodeGroups );
  parser->addItemGroupsParser  ( elems.getData(), elemGroups );
  parser->addConstraintsParser ( nodeConParser );
  parser->addConstraintsParser ( elemConParser );
  parser->setMaxParts          ( maxParts );
  parser->skipUnknownTags      ( true );

  print ( System::out(),
          "Reading mesh from `", fname, "\' ...\n\n" );

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

  if ( nodeConParser->slaveDofCount() > 0 )
  {
    nodeConParser->store ( globdat );
  }

  if ( elemConParser->slaveDofCount() > 0 )
  {
    elemConParser->store ( globdat );
  }
}


JIVE_END_PACKAGE( fem )
