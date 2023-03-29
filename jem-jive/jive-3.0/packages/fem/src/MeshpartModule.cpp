
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


#include <jem/base/rtti.h>
#include <jem/base/limits.h>
#include <jem/base/System.h>
#include <jem/base/array/select.h>
#include <jem/base/array/operators.h>
#include <jem/base/array/utilities.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalInputException.h>
#include <jem/base/IllegalArgumentException.h>
#include <jem/io/utilities.h>
#include <jem/io/FileWriter.h>
#include <jem/io/GzipFileWriter.h>
#include <jem/mp/Context.h>
#include <jem/numeric/sparse/matmul.h>
#include <jem/numeric/sparse/select.h>
#include <jem/numeric/sparse/SparseUtils.h>
#include <jem/numeric/sparse/SparseMatrix.h>
#include <jem/numeric/sparse/SparseFilters.h>
#include <jem/util/Timer.h>
#include <jem/util/Properties.h>
#include <jem/xml/print.h>
#include <jem/xml/utilities.h>
#include <jive/util/utilities.h>
#include <jive/util/ItemMap.h>
#include <jive/util/DataParser.h>
#include <jive/util/StdGroupSet.h>
#include <jive/mp/Globdat.h>
#include <jive/graph/declare.h>
#include <jive/graph/WGraph.h>
#include <jive/graph/Partitioner.h>
#include <jive/app/ModuleFactory.h>
#include <jive/fem/typedefs.h>
#include <jive/fem/Names.h>
#include <jive/fem/Globdat.h>
#include <jive/fem/ElementSet.h>
#include <jive/fem/ElementGroup.h>
#include <jive/fem/MeshpartModule.h>


JEM_DEFINE_CLASS( jive::fem::MeshpartModule );


JIVE_BEGIN_PACKAGE( fem )


using jem::newInstance;
using jem::System;
using jem::IllegalInputException;
using jive::util::joinNames;


//=======================================================================
//   class MeshpartModule::Utils_
//=======================================================================


class MeshpartModule::Utils_
{
 public:

  static SparseIdxMatrix    getElemAdj

    ( const ElementSet&       elems,
      const IdxVector&        elemWeights,
      idx_t                   minDegree );

};


//-----------------------------------------------------------------------
//   getElemGraph
//-----------------------------------------------------------------------


SparseIdxMatrix MeshpartModule::Utils_::getElemAdj

  ( const ElementSet&  elems,
    const IdxVector&   elemWeights,
    idx_t              minDegree )

{
  using jem::numeric::matmul;
  using jem::numeric::SparseMatrix;
  using jem::numeric::SparseFilters;
  using jive::util::Topology;

  Topology   elemTopo = elems.toMatrix ();

  IdxVector  ones     ( elemTopo.nonZeroCount() );

  ones = 1_idx;

  SparseIdxMatrix  elemAdj  ( elemTopo, ones );

  return matmul ( elemAdj, elemAdj.transpose(),
                  ( ! SparseFilters<idx_t>::diagonal() &&
                      SparseFilters<idx_t>::greater( minDegree - 1) ) );
}

//=======================================================================
//   class MeshpartModule
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  MeshpartModule::TYPE_NAME = "Meshpart";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


MeshpartModule::MeshpartModule ( const String& name ) :

  Super ( name )

{}


MeshpartModule::~MeshpartModule ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


app::Module::Status MeshpartModule::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  using jem::maxOf;
  using jem::toObject;
  using jem::util::Timer;
  using jive::util::expandString;
  using jive::graph::newPartitioner;

  Ref<MPContext>    mpx = mp::Globdat::getMPContext ( globdat );

  if ( mpx->myRank() != 0 )
  {
    return DONE;
  }

  Writer&           out       = jem::System::out ();
  const String      myContext = getContext       ();

  ElementSet        elems     =

    ElementSet::get ( globdat, myContext );

  const idx_t       elemCount = elems.size ();

  Properties        myConf    = conf .makeProps ( myName_ );
  Properties        myProps   = props.findProps ( myName_ );

  Ref<Partitioner>  parter;

  IdxVector         elemMap     ( elemCount );
  IdxVector         elemWeights ( elemCount );

  StringVector      domains;
  String            name;
  String            mapFile;

  Timer             timer;

  idx_t             blockCount;
  idx_t             domCount;
  idx_t             minDegree;


  blockCount = mpx->size ();
  domCount   = 1;
  minDegree  = 1;

  myProps.find ( mapFile,    PropNames::MAP_FILE );
  myProps.find ( minDegree,  PropNames::MIN_DEGREE );
  myProps.find ( blockCount, PropNames::BLOCK_COUNT,
                 1,          maxOf( blockCount ) );

  if ( myProps.find( domains, PropNames::DOMAINS ) )
  {
    domCount = domains.size ();

    for ( idx_t idom = 0; idom < domCount; idom++ )
    {
      ElementGroup  group =

        ElementGroup::get ( domains[idom], elems,
                            globdat, myContext );

      elemMap[group.getIndices()] = idom;
    }
  }

  myConf.set ( PropNames::MAP_FILE,    mapFile    );
  myConf.set ( PropNames::MIN_DEGREE,  minDegree  );
  myConf.set ( PropNames::BLOCK_COUNT, blockCount );
  myConf.set ( PropNames::DOMAINS,     domains    );

  graph::declarePartitioners ();

  getElemWeights ( elemWeights, elems );

  mapFile = expandString   ( mapFile, globdat );
  name    = joinNames      ( myName_, PropNames::PARTITIONER );
  parter  = newPartitioner ( name, conf, props, blockCount );

  print ( out, "Partitioning mesh into ",
          blockCount, " blocks ...\n" );

  timer.start ();

  if ( domCount <= 1 )
  {
    partition      ( elemMap, elemWeights, elems,
                     *parter, blockCount, minDegree );
  }
  else
  {
    multiPartition ( elemMap, elemWeights, elems,
                     *parter, blockCount, minDegree );
  }

  timer.stop  ();

  print ( out, "Ready in ", timer.toDouble(), " seconds\n" );

  print ( out, "\n  unbalance ....... ",
          getUnbalance( elemMap, elemWeights ) );
  print ( out, "\n  border nodes .... ",
          countBorderNodes( elemMap, elems ), "\n\n" );

  if ( mapFile.size() )
  {
    print ( out, "Writing block -> element map to `",
            mapFile, "\' ...\n" );

    writeElemMap ( mapFile, elemMap, elems );
  }

  globdat.set ( Globdat::ELEM_BLOCK_MAP, toObject( elemMap ) );

  return DONE;
}


//-----------------------------------------------------------------------
//   partition (without element weights)
//-----------------------------------------------------------------------


void MeshpartModule::partition

  ( const IdxVector&   elemMap,
    const ElementSet&  elems,
    Partitioner&       parter,
    idx_t              blockCount,
    idx_t              minDegree )

{
  IdxVector        elemWeights ( elems.size() );

  getElemWeights ( elemWeights, elems );

  partition      ( elemMap,
                   elemWeights,
                   elems,
                   parter,
                   blockCount,
                   minDegree );
}


//-----------------------------------------------------------------------
//   partition (with element weights)
//-----------------------------------------------------------------------


void MeshpartModule::partition

  ( const IdxVector&   elemMap,
    const IdxVector&   elemWeights,
    const ElementSet&  elems,
    Partitioner&       parter,
    idx_t              blockCount,
    idx_t              minDegree )

{
  using jive::graph::toGraph;

  JEM_PRECHECK ( elems                             &&
                 blockCount     >= 0               &&
                 elemMap.size() == elems.size()    &&
                 elemMap.size() == elemWeights.size() );

  SparseIdxMatrix  elemAdj  =

    Utils_::getElemAdj ( elems, elemWeights, minDegree );

  parter.partition ( elemMap,
                     toGraph ( elemAdj, elemWeights ),
                     blockCount );
}


//-----------------------------------------------------------------------
//   multiPartition
//-----------------------------------------------------------------------


void MeshpartModule::multiPartition

  ( const IdxVector&   elemMap,
    const IdxVector&   elemWeights,
    const ElementSet&  elems,
    Partitioner&       parter,
    idx_t              blockCount,
    idx_t              minDegree )

{
  using jem::min;
  using jem::max;
  using jem::numeric::select;
  using jem::numeric::SparseUtils;
  using jive::graph::toGraph;

  JEM_PRECHECK ( elems                             &&
                 blockCount     >= 0               &&
                 elemMap.size() == elems.size()    &&
                 elemMap.size() == elemWeights.size() );

  if ( elemMap.size() == 0 )
  {
    return;
  }

  JEM_PRECHECK ( min( elemMap ) >= 0 );

  SparseIdxMatrix  elemAdj =

    Utils_::getElemAdj ( elems, elemWeights, minDegree );

  const idx_t  domCount    = 1 + max ( elemMap );
  const idx_t  elemCount   = elemMap.size ();

  IdxVector    domOffsets  ( domCount + 1 );
  IdxVector    elemIndices ( elemCount );

  idx_t        restCount   = blockCount;
  idx_t        restWeight  = 0;


  // Group the elements per domain.

  domOffsets = 0;

  for ( idx_t ielem = 0; ielem < elemCount; ielem++ )
  {
    domOffsets[elemMap[ielem]]++;
  }

  SparseUtils::sumOffsets ( domOffsets );

  for ( idx_t ielem = 0; ielem < elemCount; ielem++ )
  {
    idx_t  idom = elemMap[ielem];

    elemIndices[domOffsets[idom]++] = ielem;
  }

  SparseUtils::shiftOffsets ( domOffsets );

  for ( idx_t ielem = 0; ielem < elemCount; ielem++ )
  {
    restWeight += elemWeights[ielem];
  }

  elemMap    = -1_idx;
  blockCount = 0;

  for ( idx_t idom = 0; idom < domCount; idom++ )
  {
    if ( restWeight <= 0 || restCount <= 1 )
    {
      break;
    }

    idx_t  i = domOffsets[idom];
    idx_t  j = domOffsets[idom + 1];

    const idx_t  domElemCount   = j - i;

    IdxVector    domElems       = elemIndices[slice(i,j)];

    IdxVector    domElemMap     ( domElemCount );
    IdxVector    domElemWeights ( domElemCount );

    idx_t        maxBlockCount  = 0;
    idx_t        domBlockCount  = 0;
    idx_t        domWeight      = 0;


    for ( i = 0; i < domElemCount; i++ )
    {
      idx_t  ielem = domElems[i];

      domWeight        += elemWeights[ielem];
      domElemWeights[i] = elemWeights[ielem];
    }

    if ( domWeight == 0 )
    {
      continue;
    }

    maxBlockCount = restCount - domCount + idom + 1;
    domBlockCount = (idx_t) ((domWeight * restCount) / restWeight);
    domBlockCount = min ( domBlockCount, maxBlockCount );
    domBlockCount = max ( domBlockCount, 1_idx );

    parter.partition (
      domElemMap,
      toGraph (
        select ( elemAdj, domElems, domElems ),
        domElemWeights
      ),
      domBlockCount
    );

    elemMap[domElems] = domElemMap + blockCount;

    blockCount += domBlockCount;
    restCount  -= domBlockCount;
    restWeight -= domWeight;
  }

  // Assign all remaining elements to the last mesh block.

  for ( idx_t ielem = 0; ielem < elemCount; ielem++ )
  {
    if ( elemMap[ielem] < 0 )
    {
      elemMap[ielem] = blockCount;
    }
  }
}


//-----------------------------------------------------------------------
//   readElemMap
//-----------------------------------------------------------------------


void MeshpartModule::readElemMap

  ( const String&       fname,
    const IdxVector&    elemMap,
    const ElementSet&   elems,
    const Ref<Writer>&  logger )

{
  using jive::util::DataParser;
  using jive::util::XGroupSet;
  using jive::util::StdGroupSet;

  JEM_PRECHECK ( elems && elems.size() == elemMap.size() );

  const idx_t      elemCount = elemMap.size ();

  Ref<DataParser>  parser;
  Ref<XGroupSet>   blocks;
  IdxVector        ielems;

  String           context;

  idx_t            blockCount;


  context = String::format ( "element map file `%s\'", fname );

  blocks  = newInstance<StdGroupSet> ( "blockElemMap",
                                       "block",
                                       elems.getData() );

  parser  = newInstance<DataParser> ();

  parser->addGroupSetParser (
    blocks, -1,
    getElemMapTag_ ( elems )
  );

  parser->skipUnknownTags ( true );
  parser->parseFile       ( fname, logger );

  if ( logger )
  {
    print ( *logger, '\n' );
  }

  ielems.resize ( blocks->maxGroupSize() );

  elemMap    = -1;
  blockCount = blocks->size ();

  for ( idx_t ib = 0; ib < blockCount; ib++ )
  {
    idx_t  iblock = blocks->getItemID ( ib );

    if ( iblock < 0 )
    {
      throw IllegalInputException (
        context,
        String::format ( "invalid block number: %d", iblock )
      );
    }

    idx_t  ielemCount = blocks->getGroupMembers ( ielems, ib );

    for ( idx_t ie = 0; ie < ielemCount; ie++ )
    {
      elemMap[ielems[ie]] = iblock;
    }
  }

  for ( idx_t ie = 0; ie < elemCount; ie++ )
  {
    if ( elemMap[ie] < 0 )
    {
      throw IllegalInputException (
        context,
        String::format (
          "element %d has not been mapped to a block",
          elems.getElemID ( ie )
        )
      );
    }
  }
}


//-----------------------------------------------------------------------
//   writeElemMap (given a file name)
//-----------------------------------------------------------------------


void MeshpartModule::writeElemMap

  ( const String&      fname,
    const IdxVector&   elemMap,
    const ElementSet&  elems )

{
  using jem::io::FileWriter;
  using jem::io::GzipFileWriter;

  Ref<PrintWriter>  prn;
  Ref<Writer>       out;


  if ( util::isGzipFile( fname ) )
  {
    out = newInstance<GzipFileWriter> ( fname );
  }
  else
  {
    out = newInstance<FileWriter>     ( fname );
  }

  prn = newInstance<PrintWriter> ( out );

  writeElemMap ( *prn, elemMap, elems );

  prn->flush ();
}


//-----------------------------------------------------------------------
//   writeElemMap (given a PrintWriter)
//-----------------------------------------------------------------------


void MeshpartModule::writeElemMap

  ( PrintWriter&       out,
    const IdxVector&   elemMap,
    const ElementSet&  elems )

{
  using jem::io::space;
  using jem::xml::startTag;
  using jem::xml::endTag;
  using jem::numeric::SparseUtils;
  using jive::util::printIntList;
  using jive::util::ItemMap;

  JEM_PRECHECK ( elems && elems.size() == elemMap.size() );

  const idx_t  elemCount = elemMap.size ();

  String       tagName   = getElemMapTag_ ( elems );

  ItemMap*     imap      = elems.getData()->getItemMap ();

  IdxVector    blockOffsets;
  IdxVector    elemIndices;
  IdxVector    elemIDs;

  idx_t        blockCount;


  if ( elemCount == 0 )
  {
    blockCount = 0;
  }
  else
  {
    if ( jem::min( elemMap ) < 0 )
    {
      throw jem::IllegalArgumentException (
        JEM_FUNC,
        "invalid element map: negative block index"
      );
    }

    blockCount = 1 + max ( elemMap );
  }

  blockOffsets.resize ( blockCount + 1 );
  elemIndices .resize ( elemCount );
  elemIDs     .resize ( elemCount );

  blockOffsets = 0;

  for ( idx_t ielem = 0; ielem < elemCount; ielem++ )
  {
    blockOffsets[elemMap[ielem]]++;
  }

  SparseUtils::sumOffsets ( blockOffsets );

  for ( idx_t ielem = 0; ielem < elemCount; ielem++ )
  {
    idx_t  iblock = elemMap[ielem];

    elemIndices[blockOffsets[iblock]++] = ielem;
  }

  SparseUtils::shiftOffsets ( blockOffsets );

  out.pushSettings ();

  out.nformat.setIntegerWidth (
    jem::io::getWidthOf ( elemCount )
  );

  imap->getItemIDs ( elemIDs, elemIndices );

  print ( out, startTag( tagName ) );

  for ( idx_t iblock = 0; iblock < blockCount; iblock++ )
  {
    idx_t  i = blockOffsets[iblock];
    idx_t  j = blockOffsets[iblock + 1];

    if ( iblock > 0 )
    {
      out.printLine ();
    }

    print        ( out, iblock, space );
    printIntList ( out, elemIDs[slice(i,j)] );
    print        ( out, ';' );
  }

  print ( out, endTag( tagName ) );

  out.popSettings  ();
}


//-----------------------------------------------------------------------
//   getElemWeights
//-----------------------------------------------------------------------


void MeshpartModule::getElemWeights

  ( const IdxVector&   weights,
    const ElementSet&  elems )

{
  using jem::max;

  JEM_PRECHECK ( elems && elems.size() == weights.size() );

  const idx_t  elemCount = weights.size ();

  for ( idx_t ielem = 0; ielem < elemCount; ielem++ )
  {
    weights[ielem] = max ( 1_idx,
                           elems.getElemNodeCount(ielem) );
  }
}


//-----------------------------------------------------------------------
//   getUnbalance
//-----------------------------------------------------------------------


double MeshpartModule::getUnbalance

  ( const IdxVector&  elemMap,
    const IdxVector&  elemWeights )

{
  using jem::max;

  JEM_PRECHECK ( elemMap.size() == elemWeights.size() );

  const idx_t  elemCount = elemMap.size ();

  IdxVector    blockWeights;

  idx_t        blockCount;


  if ( elemMap.size() )
  {
    blockCount = 1 + max ( elemMap );
  }
  else
  {
    blockCount = 1;
  }

  blockWeights.resize ( blockCount );

  blockWeights = 0;

  for ( idx_t ielem = 0; ielem < elemCount; ielem++ )
  {
    blockWeights[elemMap[ielem]] += elemWeights[ielem];
  }

  return ( (double) blockCount * (double) max( blockWeights ) /
           (double) sum( blockWeights ) - 1.0 );
}


//-----------------------------------------------------------------------
//   countBorderNodes
//-----------------------------------------------------------------------


idx_t MeshpartModule::countBorderNodes

  ( const IdxVector&   elemMap,
    const ElementSet&  elems )

{
  using jive::util::Topology;

  JEM_PRECHECK ( elemMap.size() == elems.size() );

  Topology      meshTopo    = elems.toMatrix ();

  IdxVector     elemOffsets = meshTopo.getRowOffsets    ();
  IdxVector     nodeIndices = meshTopo.getColumnIndices ();

  const idx_t   elemCount   = meshTopo.size (0);
  const idx_t   nodeCount   = meshTopo.size (1);

  IdxVector     nodeMap     ( nodeCount );

  idx_t         count;


  nodeMap = -1;
  count   =  0;

  for ( idx_t ielem = 0; ielem < elemCount; ielem++ )
  {
    idx_t  iblock = elemMap[ielem];
    idx_t  n      = elemOffsets[ielem + 1];

    for ( idx_t i = elemOffsets[ielem]; i < n; i++ )
    {
      idx_t  inode = nodeIndices[i];

      if      ( nodeMap[inode] < 0 )
      {
        nodeMap[inode] = iblock;
      }
      else if ( nodeMap[inode] != iblock )
      {
        count++;
      }
    }
  }

  return count;
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<app::Module> MeshpartModule::makeNew

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


void MeshpartModule::declare ()
{
  using jive::app::ModuleFactory;

  ModuleFactory::declare ( TYPE_NAME,  & makeNew );
  ModuleFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   getElemMapTag_
//-----------------------------------------------------------------------


String MeshpartModule::getElemMapTag_ ( const ElementSet& elems )
{
  return jem::xml::makeTag (
    elems.getData()->getItemName() + "Map"
  );
}


JIVE_END_PACKAGE( fem )
