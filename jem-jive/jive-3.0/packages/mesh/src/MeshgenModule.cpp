
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


#include <jem/base/limits.h>
#include <jem/base/System.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalInputException.h>
#include <jem/io/utilities.h>
#include <jem/io/FileWriter.h>
#include <jem/io/GzipFileWriter.h>
#include <jem/io/PrintWriter.h>
#include <jem/util/None.h>
#include <jem/util/Properties.h>
#include <jem/util/HashDictionary.h>
#include <jem/util/DictionaryEnumerator.h>
#include <jem/xml/print.h>
#include <jive/util/utilities.h>
#include <jive/util/StorageMode.h>
#include <jive/util/Assignable.h>
#include <jive/util/ItemGroup.h>
#include <jive/app/ModuleFactory.h>
#include <jive/mesh/declare.h>
#include <jive/mesh/Mesh.h>
#include <jive/mesh/Names.h>
#include <jive/mesh/Geometry.h>
#include <jive/mesh/MeshgenModule.h>


JEM_DEFINE_CLASS( jive::mesh::MeshgenModule );


JIVE_BEGIN_PACKAGE( mesh )


using jem::newInstance;
using jem::System;
using jem::util::Dictionary;
using jem::util::HashDictionary;
using jive::util::joinNames;
using jive::util::ItemGroup;


//=======================================================================
//   class MeshgenModule::Utils_
//=======================================================================


class MeshgenModule::Utils_
{
 public:

  static void           copyGroups

    ( Dictionary&         target,
      const Dictionary&   source );

  static void           saveMesh

    ( const Properties&   conf,
      const Properties&   props,
      const Properties&   globdat,
      const ElementSet&   elems,
      const Dictionary&   nodeGroups,
      const Dictionary&   elemGroups );

};


//-----------------------------------------------------------------------
//   copyGroups
//-----------------------------------------------------------------------


void MeshgenModule::Utils_::copyGroups

  ( Dictionary&         target,
    const Dictionary&   source )

{
  using jem::util::DictEnum;

  Ref<DictEnum>  e;


  for ( e = source.enumerate(); ! e->atEnd(); e->toNext() )
  {
    target.insert ( e->getKey(), e->getValue() );
  }
}


//-----------------------------------------------------------------------
//   saveMesh
//-----------------------------------------------------------------------


void MeshgenModule::Utils_::saveMesh

  ( const Properties&   conf,
    const Properties&   props,
    const Properties&   globdat,
    const ElementSet&   elems,
    const Dictionary&   nodeGroups,
    const Dictionary&   elemGroups )

{
  using jem::io::NumberFormat;
  using jem::io::Writer;
  using jem::io::PrintWriter;
  using jem::io::FileWriter;
  using jem::io::GzipFileWriter;
  using jem::xml::startTag;
  using jem::xml::endTag;

  NodeSet           nodes = elems.getNodes ();

  Ref<PrintWriter>  out;
  Ref<Writer>       file;

  String            fname;
  String            tag;

  int               prec;
  int               width;
  idx_t             n;


  prec = 8;

  props.get  ( fname, PropNames::FILE );
  props.find ( prec,  PropNames::PRECISION, 1, 16 );

  conf.set ( PropNames::FILE,      fname );
  conf.set ( PropNames::PRECISION, prec  );

  fname = util::expandString ( fname, globdat );

  if ( util::isGzipFile( fname ) )
  {
    file = newInstance<GzipFileWriter> ( fname );
  }
  else
  {
    file = newInstance<FileWriter>     ( fname );
  }

  out   = newInstance<PrintWriter> ( file );
  n     = jem::max ( nodes.size(), elems.size() );
  width = jem::io::getWidthOf ( n );

  out->nformat.setFlags          ( NumberFormat::SCIENTIFIC );
  out->nformat.setIntegerWidth   ( width    );
  out->nformat.setFloatWidth     ( prec + 7 );
  out->nformat.setFractionDigits ( prec     );

  print ( System::out(), "Writing mesh to `", fname, "\' ...\n" );

  tag = nodes.getTagName ();

  print ( *out, startTag( tag ), nodes, endTag( tag ) );

  tag = elems.getTagName ();

  print ( *out, startTag( tag ), elems, endTag( tag ) );

  ItemGroup::printSome ( *out, nodeGroups );
  ItemGroup::printSome ( *out, elemGroups );
}


//=======================================================================
//   class MeshgenModule
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  MeshgenModule::TYPE_NAME = "Meshgen";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


MeshgenModule::MeshgenModule ( const String& name ) :

  Super ( name )

{}


MeshgenModule::~MeshgenModule ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


Module::Status MeshgenModule::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  using jem::minOf;
  using jem::maxOf;
  using jem::util::isNone;
  using jem::util::None;
  using jive::util::getStorageMode;
  using jive::util::StorageMode;
  using jive::util::Assignable;
  using jive::fem::newXNodeSet;
  using jive::fem::newXElementSet;

  Assignable
    <XNodeSet>      nodes;
  Assignable
    <XElementSet>   elems;

  Properties        myConf  = conf .makeProps ( myName_ );
  Properties        myProps = props.getProps  ( myName_ );

  StorageMode       smode   = util::DEFAULT_STORAGE;

  Ref<Geometry>     geom;
  Geometry::Params  params;

  Ref<Object>       obj;


  getStorageMode ( smode,   conf,   props );
  getStorageMode ( smode, myConf, myProps );

  nodes = XNodeSet    :: find ( globdat );
  elems = XElementSet :: find ( globdat );

  if ( ! nodes )
  {
    nodes = newXNodeSet ( "nodes", smode );

    nodes.store ( globdat );
  }

  if ( ! elems )
  {
    elems = newXElementSet ( "elements", nodes, smode );

    elems.store ( globdat );
  }

  params.nodeGroups = newInstance<HashDictionary> ();
  params.elemGroups = newInstance<HashDictionary> ();

  myProps.find ( params.elemSize,
                 PropNames::ELEM_SIZE,
                 minOf ( params.elemSize ),
                 maxOf ( params.elemSize ) );

  myConf.set   ( PropNames::ELEM_SIZE, params.elemSize );

  declareGeometries ();

  geom = newGeometry ( joinNames ( myName_, PropNames::GEOMETRY ),
                       conf, props, globdat );

  geom->configure ( props );
  geom->getConfig ( conf  );

  if ( nodes.rank() != geom->rank() &&
       nodes.rank() >  0            &&
       geom->rank() >  0 )
  {
    throw jem::IllegalInputException (
      getContext (),
      String::format (
        "node rank (%d) and geometry rank (%d) do not match",
        nodes.rank (),
        geom->rank ()
      )
    );
  }

  print ( System::out(), "Generating mesh with element size ",
          params.elemSize, " ...\n" );

  geom->makeMesh ( XMesh( elems ), params );

  nodes.trimToSize ();
  elems.trimToSize ();

  if ( ! myProps.find( obj, PropNames::OUTPUT ) || isNone( obj ) )
  {
    myConf.set ( PropNames::OUTPUT, None::getInstance() );
  }
  else
  {
    Utils_::saveMesh ( myConf .makeProps ( PropNames::OUTPUT ),
                       myProps.getProps  ( PropNames::OUTPUT ),
                       globdat,
                       elems,
                       *params.nodeGroups,
                       *params.elemGroups );
  }

  Utils_::copyGroups (
    * ItemGroup::getFor ( nodes.getData(), globdat ),
    * params.nodeGroups
  );

  Utils_::copyGroups (
    * ItemGroup::getFor ( elems.getData(), globdat ),
    * params.elemGroups
  );

  print ( System::out(), '\n' );

  return DONE;
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Module> MeshgenModule::makeNew

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


void MeshgenModule::declare ()
{
  using jive::app::ModuleFactory;

  ModuleFactory::declare ( TYPE_NAME,  & makeNew );
  ModuleFactory::declare ( CLASS_NAME, & makeNew );
}


JIVE_END_PACKAGE( mesh )
