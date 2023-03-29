
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
#include <jem/io/IOException.h>
#include <jem/util/Properties.h>
#include <jive/util/utilities.h>
#include <jive/util/Table.h>
#include <jive/util/Database.h>
#include <jive/util/ItemGroup.h>
#include <jive/util/Assignable.h>
#include <jive/fem/declare.h>
#include <jive/fem/Names.h>
#include <jive/fem/XNodeSet.h>
#include <jive/fem/XElementSet.h>
#include <jive/fem/DataParser.h>
#include <jive/fem/AutoShapeTable.h>
#include <jive/geom/ShapeTable.h>
#include <jive/geom/ShapeTableFactory.h>
#include <jive/gl/Names.h>
#include <jive/gl/Viewdat.h>
#include <jive/gl/PluginFactory.h>
#include <jive/gl/MeshReader.h>


JEM_DEFINE_CLASS( jive::gl::MeshReader );


JIVE_BEGIN_PACKAGE( gl )


using jem::newInstance;
using jive::util::joinNames;
using jive::util::expandString;
using jive::util::Assignable;
using jive::util::StorageMode;
using jive::fem::XNodeSet;
using jive::fem::XElementSet;
using jive::geom::ShapeTable;


//=======================================================================
//   class MeshReader::RunData_
//=======================================================================


class MeshReader::RunData_ : public jem::Collectable
{
 public:

  inline                    RunData_    ();

  void                      initMesh

    ( const String&           name,
      StorageMode             smode,
      const Properties&       globdat );

  void                      initShapes

    ( const String&           myName,
      const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );


 public:

  Assignable<XNodeSet>      nodes;
  Assignable<XElementSet>   elems;
  Ref<ShapeTable>           shapes;

  String                    lastFile;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline MeshReader::RunData_::RunData_ ()
{}


//-----------------------------------------------------------------------
//   initMesh
//-----------------------------------------------------------------------


void MeshReader::RunData_::initMesh

  ( const String&      name,
    StorageMode        smode,
    const Properties&  globdat )

{
  using jive::fem::newXNodeSet;
  using jive::fem::newXElementSet;

  String  path;


  path  = joinNames      ( name, "nodes" );
  nodes = XNodeSet::find ( path, globdat );

  if ( ! nodes )
  {
    nodes = newXNodeSet ( path, smode );

    nodes.store ( path, globdat );
  }

  path  = joinNames         ( name, "elements" );
  elems = XElementSet::find ( path, globdat );

  if ( elems == nullptr || elems.getNodes() != nodes )
  {
    elems = newXElementSet ( path, nodes, smode );

    elems.store ( path, globdat );
  }
}


//-----------------------------------------------------------------------
//   initShapes
//-----------------------------------------------------------------------


void MeshReader::RunData_::initShapes

  ( const String&      myName,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  using jive::fem::declareShapeTables;
  using jive::fem::AutoShapeTable;
  using jive::geom::ShapeTableFactory;

  shapes = ShapeTable::find ( elems.getData(), globdat );

  if ( shapes )
  {
    // Do not manage the shape table if it already existed.

    shapes = nullptr;
  }
  else
  {
    String  name = joinNames ( myName, PropNames::SHAPE_TABLE );

    declareShapeTables ();

    if ( props.contains( name ) )
    {
      shapes = ShapeTableFactory::newInstance ( name,
                                                conf, props,
                                                elems.getData(),
                                                globdat );
    }
    else
    {
      conf.makeProps(name).set ( "type", "Auto" );

      shapes = AutoShapeTable::makeNew ( name,
                                         conf,
                                         Properties(),
                                         elems.getData(),
                                         globdat );
    }

    // Note that the shape table may be NULL if the rank of the mesh
    // is not supported.

    if ( shapes )
    {
      shapes->store     ( globdat );
      shapes->configure ( props, globdat );
      shapes->getConfig ( conf );
    }
  }
}


//=======================================================================
//   class MeshReader
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  MeshReader::TYPE_NAME = "MeshReader";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


MeshReader::MeshReader ( const String& name ) :

  Super ( name )

{
  tableFilter_ = "*";
  dbaseFilter_ = "*";
  maxParts_    = 0;
}


MeshReader::~MeshReader ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


void MeshReader::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  viewdat,
    const AppContext&  apx )

{
  using jive::util::getStorageMode;

  rundat_ = nullptr;

  Properties     globdat = viewdat.getProps ( Viewdat::GLOBDAT );

  Properties     myConf  = conf .makeProps ( myName_ );
  Properties     myProps = props.findProps ( myName_ );

  StorageMode    smode   = util::COMPACT_STORAGE;

  Ref<RunData_>  newdat  = newInstance<RunData_> ();

  String         fname   = expandString ( fileName_, globdat );


  getStorageMode     ( smode,   conf,   props );
  getStorageMode     ( smode, myConf, myProps );

  newdat->initMesh   ( meshName_, smode,       globdat );
  readData_          ( *newdat,   fname,       globdat );
  newdat->initShapes ( myName_,   conf, props, globdat );
  rundat_.swap       ( newdat );
}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void MeshReader::update

  ( const Properties&  viewdat,
    const AppContext&  apx )

{
  if ( rundat_ && fileName_.size() )
  {
    Properties  globdat = viewdat.getProps ( Viewdat::GLOBDAT );
    String      fname   = expandString     ( fileName_, globdat );


    if ( fname != rundat_->lastFile )
    {
      readData_ ( *rundat_, fname, globdat );
    }
  }
}


//-----------------------------------------------------------------------
//   shutdown
//-----------------------------------------------------------------------


void MeshReader::shutdown

  ( const Properties&  viewdat,
    const AppContext&  apx )

{
  rundat_ = nullptr;
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void MeshReader::configure

  ( const Properties&  props,
    const Properties&  viewdat,
    const AppContext&  apx )

{
  if ( props.contains( myName_ ) )
  {
    Properties  myProps = props.findProps ( myName_ );

    myProps.find ( meshName_, PropNames::MESH_NAME );

    if ( myProps.find( fileName_, PropNames::FILE_NAME ) )
    {
      if ( rundat_ )
      {
        rundat_->lastFile = "";
      }
    }

    myProps.find ( tableFilter_, fem::PropNames::TABLE_FILTER );
    myProps.find ( dbaseFilter_, fem::PropNames::DBASE_FILTER );
    myProps.find ( maxParts_,    fem::PropNames::MAX_PARTS,
                   0,            10000 );
  }

  if ( rundat_ && rundat_->shapes )
  {
    Properties  globdat = viewdat.getProps ( Viewdat::GLOBDAT );

    rundat_->shapes->configure ( props, globdat );
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void MeshReader::getConfig ( const Properties& conf ) const
{
  Properties  myConf = conf.makeProps ( myName_ );

  myConf.set ( PropNames::MESH_NAME,         meshName_    );
  myConf.set ( PropNames::FILE_NAME,         fileName_    );
  myConf.set ( fem::PropNames::TABLE_FILTER, tableFilter_ );
  myConf.set ( fem::PropNames::DBASE_FILTER, dbaseFilter_ );
  myConf.set ( fem::PropNames::MAX_PARTS,    maxParts_    );

  if ( rundat_ && rundat_->shapes )
  {
    rundat_->shapes->getConfig ( conf );
  }
}


//-----------------------------------------------------------------------
//   getNodes
//-----------------------------------------------------------------------


NodeSet MeshReader::getNodes () const
{
  if ( rundat_ )
  {
    return rundat_->nodes;
  }
  else
  {
    return NodeSet ();
  }
}


//-----------------------------------------------------------------------
//   getElements
//-----------------------------------------------------------------------


ElementSet MeshReader::getElements () const
{
  if ( rundat_ )
  {
    return rundat_->elems;
  }
  else
  {
    return ElementSet ();
  }
}


//-----------------------------------------------------------------------
//   setMeshName
//-----------------------------------------------------------------------


void MeshReader::setMeshName ( const String& name )
{
  meshName_ = name;
}


//-----------------------------------------------------------------------
//   getMeshName
//-----------------------------------------------------------------------


String MeshReader::getMeshName () const
{
  return meshName_;
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void MeshReader::declare ()
{
  PluginFactory::declare ( TYPE_NAME,  & makeNew );
  PluginFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Plugin> MeshReader::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props )

{
  return newInstance<Self> ( name );
}


//-----------------------------------------------------------------------
//   readData_
//-----------------------------------------------------------------------


void MeshReader::readData_

  ( RunData_&          rundat,
    const String&      fname,
    const Properties&  globdat )

{
  using jem::System;
  using jem::io::Writer;
  using jem::io::IOException;
  using jem::util::Dict;
  using jive::util::Database;
  using jive::fem::DataParser;

  Writer&          info =   System::info ( myName_ );

  Ref<DataParser>  parser;

  Ref<ItemSet>     items[2];
  Ref<Dict>        groups;
  Ref<Dict>        tables;
  Ref<Dict>        dbases;


  parser = newInstance<DataParser> ();

  parser->setMaxParts     ( maxParts_ );
  parser->skipUnknownTags ( true );

  rundat.nodes.clear ();
  rundat.elems.clear ();

  items[0] = rundat.nodes.getData ();
  items[1] = rundat.elems.getData ();

  parser->addNodesParser    ( rundat.nodes );
  parser->addElementsParser ( rundat.elems );

  for ( int i = 0; i < 2; i++ )
  {
    groups = ItemGroup::getFor  ( items[i], globdat );

    parser->addItemGroupsParser ( items[i], groups );

    if ( tableFilter_.size() )
    {
      tables = Table::getFor  ( items[i], globdat );

      parser->addTablesParser ( items[i], tables, groups,
                                tableFilter_ );
    }

    if ( dbaseFilter_.size() )
    {
      dbases = Database::getFor ( items[i], globdat );

      parser->addDbasesParser   ( items[i], dbases, groups,
                                  dbaseFilter_ );
    }
  }

  print ( info, "Reading data from `", fname, "\' ...\n\n" );

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

  rundat.lastFile = fname;
}


JIVE_END_PACKAGE( gl )
