
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
#include <jem/base/array/utilities.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalInputException.h>
#include <jem/base/IllegalArgumentException.h>
#include <jem/io/utilities.h>
#include <jem/io/File.h>
#include <jem/io/PrintWriter.h>
#include <jem/io/IOException.h>
#include <jem/mp/Buffer.h>
#include <jem/mp/Context.h>
#include <jem/util/Properties.h>
#include <jem/util/Dictionary.h>
#include <jem/xml/utilities.h>
#include <jive/util/utilities.h>
#include <jive/util/Table.h>
#include <jive/util/ItemMap.h>
#include <jive/util/ItemGroup.h>
#include <jive/util/Database.h>
#include <jive/util/DataParser.h>
#include <jive/util/StorageMode.h>
#include <jive/util/SparseDofSpace.h>
#include <jive/util/Constraints.h>
#include <jive/util/ConstraintsParser.h>
#include <jive/mp/scatter.h>
#include <jive/mp/utilities.h>
#include <jive/mp/Globdat.h>
#include <jive/mp/ItemMask.h>
#include <jive/graph/declare.h>
#include <jive/graph/Partitioner.h>
#include <jive/app/ModuleFactory.h>
#include <jive/fem/typedefs.h>
#include <jive/fem/Names.h>
#include <jive/fem/Globdat.h>
#include <jive/fem/MeshBlock.h>
#include <jive/fem/MeshCutter.h>
#include <jive/fem/XBoundarySet.h>
#include <jive/fem/InvalidBlocksException.h>
#include <jive/fem/MeshpartModule.h>
#include <jive/fem/PartitionModule.h>


JEM_DEFINE_CLASS( jive::fem::PartitionModule );


JIVE_BEGIN_PACKAGE( fem )


using jem::newInstance;
using jem::System;
using jem::IllegalInputException;
using jem::io::space;
using jem::io::Writer;
using jem::numeric::SparseStructure;
using jem::util::Dict;
using jive::util::joinNames;
using jive::util::Table;
using jive::util::Database;


//=======================================================================
//   class PartitionModule::InitData_
//=======================================================================


class PartitionModule::InitData_
{
 public:

                            InitData_

    ( const String&           name,
      const String&           ctx,
      const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );


 public:

  const String              myName;
  const String              context;

  Properties                globConf;
  Properties                globProps;

  Writer&                   out;
  Writer&                   info;
  Writer&                   debug;

  util::StorageMode         smode;
  String                    mapFile;
  idx_t                     minDegree;
  idx_t                     overlap;
  bool                      checkBlocks;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


PartitionModule::InitData_::InitData_

  ( const String&      name,
    const String&      ctx,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat ) :

    myName    (  name ),
    context   (   ctx ),
    globConf  (  conf ),
    globProps ( props ),

    out       ( System::out()         ),
    info      ( System::info ( name ) ),
    debug     ( System::debug( name ) )

{
  using jem::maxOf;
  using jive::util::getStorageMode;

  Properties  myProps = props.findProps ( myName );
  Properties  myConf  = conf .makeProps ( myName );


  smode       = util::DEFAULT_STORAGE;
  minDegree   = 1;
  overlap     = 0;
  checkBlocks = false;

  getStorageMode ( smode,   conf,   props );
  getStorageMode ( smode, myConf, myProps );

  if ( props.find( mapFile, CASE_NAME ) )
  {
    mapFile = mapFile + ".map";

    if ( jem::io::canZip() )
    {
      mapFile = mapFile + ".gz";
    }
  }

  myProps.find ( mapFile,     PropNames::MAP_FILE     );
  myProps.find ( checkBlocks, PropNames::CHECK_BLOCKS );
  myProps.find ( minDegree,   PropNames::MIN_DEGREE   );
  myProps.find ( overlap,     PropNames::OVERLAP,
                 0,           maxOf( overlap ) );

  myConf.set ( PropNames::MAP_FILE,     mapFile     );
  myConf.set ( PropNames::MIN_DEGREE,   minDegree   );
  myConf.set ( PropNames::CHECK_BLOCKS, checkBlocks );
  myConf.set ( PropNames::OVERLAP,      overlap     );

  mapFile = util::expandString ( mapFile, globdat );
}


//=======================================================================
//   class PartitionModule::Utils_
//=======================================================================


class PartitionModule::Utils_
{
 public:

  static void               printInfo

    ( const MeshCutter&       mcutter );

  static inline String      getElemMapTag

    ( const ElementSet&       elems );

  static inline void        storeBlock

    ( const Properties&       globdat,
      const MeshBlock&        mblock );

  static bool               loadElemMap

    ( const InitData_&        indat,
      const IdxVector&        elemMap,
      const ElementSet&       elems,
      idx_t                   blockCount );

  static bool               getConstraints

    ( const InitData_&        indat,
      SparseStructure&        conmat,
      const Ref<ItemSet>&     items,
      const Properties&       globdat );

  static Ref<MeshBlock>     makeBlocks

    ( MPContext&              mpx,
      const InitData_&        indat,
      const ElementSet&       elems,
      const Properties&       globdat );

  static void               checkBorders

    ( MPContext&              mpx,
      const InitData_&        indat,
      const MeshBlock&        mblock );

  static void               scatterGroups

    ( MPContext&              mpx,
      const InitData_&        indat,
      const ItemSetSkin&      recvItems,
      const ItemSetSkin&      sendItems,
      const Properties&       globdat );

  static void               scatterTables

    ( MPContext&              mpx,
      const InitData_&        indat,
      const ItemSetSkin&      recvItems,
      const ItemSetSkin&      sendItems,
      const Properties&       globdat );

  static void               scatterDBases

    ( MPContext&              mpx,
      const InitData_&        indat,
      const ItemSetSkin&      recvItems,
      const ItemSetSkin&      sendItems,
      const Properties&       globdat );

  static void               recvGroups

    ( MPContext&              mpx,
      const InitData_&        indat,
      const ItemSetSkin&      recvItems,
      const Properties&       globdat );

  static void               recvTables

    ( MPContext&              mpx,
      const InitData_&        indat,
      const ItemSetSkin&      recvItems,
      const Properties&       globdat );

  static void               recvDBases

    ( MPContext&              mpx,
      const InitData_&        indat,
      const ItemSetSkin&      recvItems,
      const Properties&       globdat );

  static void               scatterBounds

    ( MPContext&              mpx,
      const InitData_&        indat,
      const BoundarySet&      bounds,
      const MeshBlock&        mblock,
      const Properties&       globdat );

  static void               initRoot

    ( MPContext&              mpx,
      const InitData_&        indat,
      const Properties&       globdat );

  static void               initRest

    ( MPContext&              mpx,
      const InitData_&        indat,
      const Properties&       globdat );

};


//-----------------------------------------------------------------------
//   printInfo
//-----------------------------------------------------------------------


void PartitionModule::Utils_::printInfo

  ( const MeshCutter& mcutter )

{
  using jem::io::endl;
  using jem::io::flush;
  using jem::io::indent;
  using jem::io::outdent;

  Ref<PrintWriter>  out =

    newInstance<PrintWriter> ( & System::out() );

  MeshCutter::Info  info;


  mcutter.getInfo      ( info );

  print ( *out, endl, "Mesh partition summary:", indent, endl );
  print ( *out, info, outdent, endl, endl, flush );
}


//-----------------------------------------------------------------------
//   getElemMapTag
//-----------------------------------------------------------------------


inline String PartitionModule::Utils_::getElemMapTag

  ( const ElementSet& elems )

{
  return jem::xml::makeTag (
    elems.getData()->getItemName() + "Map"
  );
}


//-----------------------------------------------------------------------
//   storeBlock
//-----------------------------------------------------------------------


inline void PartitionModule::Utils_::storeBlock

  ( const Properties&  globdat,
    const MeshBlock&   mblock )

{
  using jive::mp::RECV_BORDERS;
  using jive::mp::SEND_BORDERS;

  mblock.nodes          .store ( globdat );
  mblock.elems          .store ( globdat );

  mblock.nodeMask      ->store ( globdat );
  mblock.elemMask      ->store ( globdat );

  mblock.recvNodeBorders.store ( RECV_BORDERS, globdat );
  mblock.sendNodeBorders.store ( SEND_BORDERS, globdat );

  mblock.recvElemBorders.store ( RECV_BORDERS, globdat );
  mblock.sendElemBorders.store ( SEND_BORDERS, globdat );
}


//-----------------------------------------------------------------------
//   loadElemMap
//-----------------------------------------------------------------------


bool PartitionModule::Utils_::loadElemMap

  ( const InitData_&   indat,
    const IdxVector&   elemMap,
    const ElementSet&  elems,
    idx_t              blockCount )

{
  using jem::io::File;

  if ( ! File::exists( indat.mapFile ) )
  {
    return false;
  }

  print ( indat.out, "Reading block -> element map from `",
          indat.mapFile, "\' ...\n" );

  bool  ok = true;

  try
  {
    MeshpartModule::readElemMap ( indat.mapFile, elemMap,
                                  elems, &indat.info );
  }
  catch ( const jem::Exception& ex )
  {
    Writer&  warn = System::warn ();

    print ( warn, '\n' );
    print ( warn, "Error reading block -> element map:\n\n" );
    print ( warn, "  context : ", ex.where (), '\n' );
    print ( warn, "  error   : ", ex.what  (), "\n\n" );

    ok = false;
  }

  if ( ok && elemMap.size() > 0 &&
       max( elemMap ) != (blockCount - 1) )
  {
    print ( System::warn(),
            "\nInvalid number of blocks specified: ",
            max ( elemMap ) + 1,
            " (should be ", blockCount, ")\n" );

    ok = false;
  }

  if ( ! ok )
  {
    print ( System::warn(),
            "(discarding block -> element map)\n\n" );
  }

  return ok;
}


//-----------------------------------------------------------------------
//   getConstraints
//-----------------------------------------------------------------------


bool PartitionModule::Utils_::getConstraints

  ( const InitData_&     indat,
    SparseStructure&     conMat,
    const Ref<ItemSet>&  items,
    const Properties&    globdat )

{
  using jive::util::Constraints;
  using jive::util::ConParser;
  using jive::util::SparseDofSpace;

  Ref<Constraints>  cons;
  Ref<DofSpace>     dofs;


  dofs = DofSpace::find ( items, globdat );

  if ( dofs )
  {
    cons = Constraints::extract ( dofs, globdat );
  }

  if ( ! cons )
  {
    Ref<ConParser>  parser;
    Ref<XDofSpace>  xdofs;

    parser = ConParser::extract ( items, globdat );

    if ( parser )
    {
      xdofs  = newInstance<SparseDofSpace> ( items );
      dofs   = xdofs;
      cons   = newInstance<Constraints>    ( dofs );

      parser->addDofsTo      ( *xdofs );
      parser->getConstraints ( *cons, indat.context );
      cons  ->compress       ();
    }
  }

  if ( cons == nullptr || cons->masterDofCount() == 0 )
  {
    return false;
  }
  else
  {
    const idx_t  dofCount    = cons->dofCount       ();
    const idx_t  slaveCount  = cons->slaveDofCount  ();
    const idx_t  masterCount = cons->masterDofCount ();

    IdxVector    slaveDofs   ( slaveCount );
    IdxVector    masterDofs  ( dofCount   );

    IdxVector    conOffsets  ( slaveCount + 1 );
    IdxVector    conIndices  ( slaveCount + masterCount );

    idx_t        iitem;
    idx_t        itype;
    idx_t        idof;
    idx_t        i, j, k, n;


    cons->getSlaveDofs ( slaveDofs );

    k = 0;

    for ( i = 0; i < slaveCount; i++ )
    {
      idof = slaveDofs[i];
      n    = cons->getMasterDofs ( masterDofs, idof );

      dofs->decodeDofIndex ( iitem, itype, idof );

      conOffsets[i]   = k;
      conIndices[k++] = iitem;

      for ( j = 0; j < n; j++ )
      {
        idof = masterDofs[j];

        dofs->decodeDofIndex ( iitem, itype, idof );

        conIndices[k++] = iitem;
      }
    }

    conOffsets[slaveCount] = k;

    conMat = SparseStructure (
      jem::shape ( slaveCount, items->size() ),
      conOffsets,
      conIndices
    );

    return true;
  }
}


//-----------------------------------------------------------------------
//   makeBlocks
//-----------------------------------------------------------------------


Ref<MeshBlock> PartitionModule::Utils_::makeBlocks

  ( MPContext&         mpx,
    const InitData_&   indat,
    const ElementSet&  elems,
    const Properties&  globdat )

{
  using jem::toValue;
  using jem::isInstance;
  using jive::graph::Partitioner;
  using jive::graph::newPartitioner;

  const int           procCount = mpx.size        ();

  NodeSet             nodes     = elems.getNodes  ();

  MeshCutter::Params  mparams   ( elems );

  Ref<Partitioner>    parter;
  Ref<MeshCutter>     mcutter;
  Ref<MeshBlock>      mblock;
  Ref<Object>         obj;

  String              name;
  bool                validMap;
  int                 rank;


  validMap  = false;

  if ( globdat.find( obj, Globdat::ELEM_BLOCK_MAP ) )
  {
    globdat.erase ( Globdat::ELEM_BLOCK_MAP );

    if ( isInstance<IdxVector>( obj ) )
    {
      mparams.elemMap.ref ( toValue<const IdxVector&>( obj ) );

      validMap = (mparams.elemMap.size() == elems.size());
    }
  }

  if ( ! validMap && indat.mapFile.size() )
  {
    validMap = loadElemMap ( indat, mparams.elemMap,
                             elems, procCount );
  }

  if ( ! validMap )
  {
    graph::declarePartitioners ();

    name   = joinNames      ( indat.myName,
                              PropNames::PARTITIONER );
    parter = newPartitioner ( name,
                              indat.globConf, indat.globProps );

    print ( indat.out, "Partitioning mesh into ",
            procCount, " blocks ...\n" );

    MeshpartModule::partition ( mparams.elemMap,
                                elems,
                                *parter,
                                procCount,
                                indat.minDegree );

    if ( indat.mapFile.size() )
    {
      print ( indat.out, "Writing block -> element map to `",
              indat.mapFile, "\' ...\n" );

      MeshpartModule::writeElemMap ( indat.mapFile,
                                     mparams.elemMap,
                                     elems );
    }

    validMap = true;
  }

  mparams.overlap = indat.overlap;

  getConstraints ( indat, mparams.nodeCons,
                   nodes.getData(), globdat );
  getConstraints ( indat, mparams.elemCons,
                   elems.getData(), globdat );

  if ( mparams.elemCons.size(0) > 0 && indat.overlap == 0 )
  {
    print ( System::warn(),
            "Element constraints require overlapping "
            "partitions;\nSetting overlap to 1\n" );

    mparams.overlap = 1;
  }

  try
  {
    mcutter = newInstance<MeshCutter> ( mparams );
  }
  catch ( const MeshCutter::Exception& ex )
  {
    throw IllegalInputException ( indat.context, ex.what() );
  }

  printInfo ( *mcutter );

  if ( indat.checkBlocks )
  {
    print ( indat.debug, "Checking mesh blocks ...\n" );

    try
    {
      checkMeshBlocks ( *mcutter );
    }
    catch ( InvalidBlocksException& ex )
    {
      ex.setContext ( indat.context );
      throw;
    }

    print ( indat.debug, "Mesh blocks are OK.\n" );
  }

  print ( indat.out, "Scattering mesh blocks ...\n" );

  for ( rank = 1; rank < procCount; rank++ )
  {
    mblock = mcutter->makeBlock ( rank, indat.smode );

    print ( indat.info,
            "  sending block to process ", rank, " ...\n" );

    sendMeshBlock ( mpx, rank, *mblock );
  }

  return mcutter->makeBlock ( 0 );
}


//-----------------------------------------------------------------------
//   checkBorders
//-----------------------------------------------------------------------


void PartitionModule::Utils_::checkBorders

  ( MPContext&        mpx,
    const InitData_&  indat,
    const MeshBlock&  mblock )

{
  print ( indat.debug, "  checking node borders ...\n" );

  mp::checkBorders ( mpx,
                     mblock.recvNodeBorders,
                     mblock.sendNodeBorders );

  if ( indat.overlap <= 0 )
  {
    return;
  }

  print ( indat.debug, "  checking element borders ...\n" );

  mp::checkBorders ( mpx,
                     mblock.recvElemBorders,
                     mblock.sendElemBorders );
}


//-----------------------------------------------------------------------
//   scatterGroups
//-----------------------------------------------------------------------


void PartitionModule::Utils_::scatterGroups

  ( MPContext&          mpx,
    const InitData_&    indat,
    const ItemSetSkin&  recvItems,
    const ItemSetSkin&  sendItems,
    const Properties&   globdat )

{
  Ref<ItemSet>  recvSet  = recvItems.getData ();
  Ref<ItemSet>  sendSet  = sendItems.getData ();

  const String  itemName = sendSet->getItemName ();

  Ref<Dict>     recvDict =

    ItemGroup::getFor ( recvSet, globdat );

  Ref<Dict>     sendDict =

    ItemGroup::getFor ( sendSet, globdat );


  print ( indat.info, "  scattering ", sendDict->size(),
          space, itemName, " groups ...\n" );

  mp::scatterGroups ( mpx, *recvDict, *sendDict, recvSet );
}


//-----------------------------------------------------------------------
//   scatterTables
//-----------------------------------------------------------------------


void PartitionModule::Utils_::scatterTables

  ( MPContext&          mpx,
    const InitData_&    indat,
    const ItemSetSkin&  recvItems,
    const ItemSetSkin&  sendItems,
    const Properties&   globdat )

{
  Ref<ItemSet>  recvSet  = recvItems.getData ();
  Ref<ItemSet>  sendSet  = sendItems.getData ();

  const String  itemName = sendSet->getItemName ();

  Ref<Dict>     recvDict = Table::getFor ( recvSet, globdat );
  Ref<Dict>     sendDict = Table::getFor ( sendSet, globdat );


  print ( indat.info, "  scattering ", sendDict->size(),
          space, itemName, " tables ...\n" );

  mp::scatterTables ( mpx, *recvDict, *sendDict, recvSet );
}


//-----------------------------------------------------------------------
//   scatterDBases
//-----------------------------------------------------------------------


void PartitionModule::Utils_::scatterDBases

  ( MPContext&          mpx,
    const InitData_&    indat,
    const ItemSetSkin&  recvItems,
    const ItemSetSkin&  sendItems,
    const Properties&   globdat )

{
  Ref<ItemSet>  recvSet  = recvItems.getData ();
  Ref<ItemSet>  sendSet  = sendItems.getData ();

  const String  itemName = sendSet->getItemName ();

  Ref<Dict>     recvDict = Database::getFor ( recvSet, globdat );
  Ref<Dict>     sendDict = Database::getFor ( sendSet, globdat );


  print ( indat.info, "  scattering ", sendDict->size(),
          space, itemName, " databases ...\n" );

  mp::scatterDBases ( mpx, *recvDict, *sendDict, recvSet );
}


//-----------------------------------------------------------------------
//   recvGroups
//-----------------------------------------------------------------------


void PartitionModule::Utils_::recvGroups

  ( MPContext&          mpx,
    const InitData_&    indat,
    const ItemSetSkin&  recvItems,
    const Properties&   globdat )

{
  Ref<ItemSet>  recvSet  = recvItems.getData    ();
  const String  itemName = recvSet->getItemName ();

  Ref<Dict>     recvDict =

    ItemGroup::getFor ( recvSet, globdat );

  mp::recvGroups ( mpx, *recvDict, recvSet );

  print ( indat.debug, "  received ", recvDict->size(),
          space, itemName, " groups from process 0\n" );
}


//-----------------------------------------------------------------------
//   recvTables
//-----------------------------------------------------------------------


void PartitionModule::Utils_::recvTables

  ( MPContext&          mpx,
    const InitData_&    indat,
    const ItemSetSkin&  recvItems,
    const Properties&   globdat )

{
  Ref<ItemSet>  recvSet  = recvItems.getData    ();
  const String  itemName = recvSet->getItemName ();

  Ref<Dict>     recvDict = Table::getFor ( recvSet, globdat );

  mp::recvTables ( mpx, *recvDict, recvSet );

  print ( indat.debug, "  received ", recvDict->size(),
          space, itemName, " tables from process 0\n" );
}


//-----------------------------------------------------------------------
//   recvDBases
//-----------------------------------------------------------------------


void PartitionModule::Utils_::recvDBases

  ( MPContext&          mpx,
    const InitData_&    indat,
    const ItemSetSkin&  recvItems,
    const Properties&   globdat )

{
  Ref<ItemSet>  recvSet  = recvItems.getData    ();
  const String  itemName = recvSet->getItemName ();

  Ref<Dict>     recvDict = Database::getFor ( recvSet, globdat );

  mp::recvDBases ( mpx, *recvDict, recvSet );

  print ( indat.debug, "  received ", recvDict->size(),
          space, itemName, " databases from process 0\n" );
}


//-----------------------------------------------------------------------
//   scatterBounds
//-----------------------------------------------------------------------


void PartitionModule::Utils_::scatterBounds

  ( MPContext&          mpx,
    const InitData_&    indat,
    const BoundarySet&  bounds,
    const MeshBlock&    mblock,
    const Properties&   globdat )

{
  XBoundarySet  myBounds = newXBoundarySet ( mblock.elems,
                                             indat.smode );

  print ( indat.out, "Scattering boundaries ...\n" );

  mp::scatterMembers ( mpx,
                       * myBounds.getData(),
                       * bounds  .getData() );

  scatterGroups      ( mpx, indat, myBounds, bounds, globdat );
  scatterTables      ( mpx, indat, myBounds, bounds, globdat );
  scatterDBases      ( mpx, indat, myBounds, bounds, globdat );

  util::Globdat::eraseDataFor ( bounds.getData(), globdat );

  myBounds.store     ( globdat );
}


//-----------------------------------------------------------------------
//   initRoot
//-----------------------------------------------------------------------


void PartitionModule::Utils_::initRoot

  ( MPContext&         mpx,
    const InitData_&   indat,
    const Properties&  globdat )

{
  using jem::io::IOException;
  using jem::mp::SendBuffer;


  BoundarySet      bounds = BoundarySet::find ( globdat );
  ElementSet       elems  = ElementSet ::get  ( globdat,
                                                indat.context );
  NodeSet          nodes  = elems.getNodes    ();

  Ref<MeshBlock>   mblock;
  int              bndFlag;


  try
  {
    mblock = makeBlocks ( mpx, indat, elems, globdat );
  }
  catch ( IOException& ex )
  {
    ex.setContext ( indat.context );
    throw;
  }

#ifndef NDEBUG

  checkBorders  ( mpx, indat, *mblock );

#endif

  print ( indat.out, "Scattering other data ...\n" );

  scatterGroups ( mpx, indat, mblock->nodes, nodes, globdat );
  scatterGroups ( mpx, indat, mblock->elems, elems, globdat );
  scatterTables ( mpx, indat, mblock->nodes, nodes, globdat );
  scatterTables ( mpx, indat, mblock->elems, elems, globdat );
  scatterDBases ( mpx, indat, mblock->nodes, nodes, globdat );
  scatterDBases ( mpx, indat, mblock->elems, elems, globdat );

  util::Globdat::eraseDataFor ( nodes.getData(), globdat );
  util::Globdat::eraseDataFor ( elems.getData(), globdat );

  storeBlock    ( globdat, *mblock );

  if ( bounds && elems == bounds.getElements() )
  {
    bndFlag = 1;

    mpx.broadcast ( SendBuffer( &bndFlag, 1 ) );

    scatterBounds ( mpx, indat, bounds, *mblock, globdat );
  }
  else
  {
    bndFlag = 0;

    mpx.broadcast ( SendBuffer( &bndFlag, 1 ) );
  }

  print ( indat.out, '\n' );
}


//-----------------------------------------------------------------------
//   initRest
//-----------------------------------------------------------------------


void PartitionModule::Utils_::initRest

  ( MPContext&         mpx,
    const InitData_&   indat,
    const Properties&  globdat )

{
  using jem::mp::RecvBuffer;

  Ref<MeshBlock>   mblock;
  int              bndFlag;


  print ( indat.debug, "Receiving data from process 0 ...\n\n" );

  mblock = recvMeshBlock ( mpx, 0 );

#ifndef NDEBUG

  checkBorders ( mpx, indat, *mblock );

#endif

  recvGroups   ( mpx, indat, mblock->nodes, globdat );
  recvGroups   ( mpx, indat, mblock->elems, globdat );
  recvTables   ( mpx, indat, mblock->nodes, globdat );
  recvTables   ( mpx, indat, mblock->elems, globdat );
  recvDBases   ( mpx, indat, mblock->nodes, globdat );
  recvDBases   ( mpx, indat, mblock->elems, globdat );

  storeBlock   ( globdat, *mblock );

  mpx.broadcast ( RecvBuffer( &bndFlag, 1 ), 0 );

  if ( bndFlag )
  {
    XBoundarySet  bounds = newXBoundarySet ( mblock->elems,
                                             indat.smode );

    mp::recvMembers ( mpx, *bounds.getData() );

    print ( indat.debug, "  received ", bounds.size(),
	    " boundaries from process 0\n" );

    recvGroups ( mpx, indat, bounds, globdat );
    recvTables ( mpx, indat, bounds, globdat );
    recvDBases ( mpx, indat, bounds, globdat );

    bounds.store ( globdat );
  }

  print ( indat.debug, '\n' );
}


//=======================================================================
//   class PartitionModule
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  PartitionModule::TYPE_NAME = "FEPartition";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


PartitionModule::PartitionModule ( const String& name ) :

  Super ( name )

{}


PartitionModule::~PartitionModule ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


app::Module::Status PartitionModule::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  Ref<MPContext>  mpx = mp::Globdat::getMPContext ( globdat );

  if ( mpx->size() > 1 )
  {
    InitData_  indat ( myName_, getContext(), conf, props, globdat );


    if ( mpx->myRank() == 0 )
    {
      Utils_::initRoot ( *mpx, indat, globdat );
    }
    else
    {
      Utils_::initRest ( *mpx, indat, globdat );
    }
  }

  return DONE;
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<app::Module> PartitionModule::makeNew

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


void PartitionModule::declare ()
{
  using jive::app::ModuleFactory;

  ModuleFactory::declare ( TYPE_NAME,  & makeNew );
  ModuleFactory::declare ( CLASS_NAME, & makeNew );
}


JIVE_END_PACKAGE( fem )
