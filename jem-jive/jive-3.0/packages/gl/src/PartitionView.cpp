
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
#include <jem/base/Class.h>
#include <jem/base/Time.h>
#include <jem/base/System.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalInputException.h>
#include <jem/base/array/select.h>
#include <jem/base/array/utilities.h>
#include <jem/io/IOException.h>
#include <jem/gl/byte.h>
#include <jem/gl/Compound.h>
#include <jem/gl/PointSampler.h>
#include <jem/gl/MessageBox.h>
#include <jem/util/Event.h>
#include <jem/util/Properties.h>
#include <jem/numeric/sparse/SparseUtils.h>
#include <jive/util/utilities.h>
#include <jive/fem/Globdat.h>
#include <jive/fem/ElementGroup.h>
#include <jive/fem/MeshpartModule.h>
#include <jive/gl/Names.h>
#include <jive/gl/Viewdat.h>
#include <jive/gl/MeshRenderer.h>
#include <jive/gl/PluginFactory.h>
#include <jive/gl/PartitionView.h>


JEM_DEFINE_CLASS( jive::gl::PartitionView );


JIVE_BEGIN_PACKAGE( gl )


using jem::newInstance;
using jem::Array;
using jem::util::Event;
using jem::gl::PointSampler;


//=======================================================================
//   class PartitionView::RunData_
//=======================================================================


class PartitionView::RunData_ : public jem::Collectable
{
 public:

  typedef RunData_          Self;
  typedef
    PointSampler::Sample    Sample;


  struct                    BlockInfo
  {
    idx_t                     elemCount;
    idx_t                     nodeCount;
    idx_t                     bnodeCount;
  };


  explicit inline           RunData_

    ( const String&           nam,
      const String&           ctx );

  void                      init

    ( const Properties&       myConf,
      const Properties&       myProps,
      const Properties&       viewdat );

  void                      update

    ( const Properties&       viewdat,
      const AppContext&       apx );

  inline void               shutdown        ();

  inline void               configure

    ( const Properties&       myProps );

  inline void               getConfig

    ( const Properties&       myProps )        const;


 public:

  const String              myName;
  const String              myContext;


 private:

  void                      pointSampled_

    ( RenderContext&          rtx,
      const Sample&           sample );

  void                      updateColors_

    ( const AppContext&       apx );

  void                      updateInfo_

    ( const IdxVector&        elmap,
      const AppContext&       apx );

  void                      deleteMap_      ();


 private:

  Event < RenderContext&,
          const Sample& >   sampleEvent_;

  Ref<MeshRenderer>         mrender_;
  Ref<PointSampler>         sampler_;

  String                    mapFile_;
  idx_t                     blockCount_;
  Array<BlockInfo>          blockInfo_;
  IdxVector                 elemBlockMap_;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline PartitionView::RunData_::RunData_

  ( const String&  nam,
    const String&  ctx ) :

    myName    ( nam ),
    myContext ( ctx )

{
  blockCount_ = 0;
}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


void PartitionView::RunData_::init

  ( const Properties&  myConf,
    const Properties&  myProps,
    const Properties&  viewdat )

{
  using jem::gl::addChild;
  using jem::gl::GfxNode;
  using jem::util::connect;
  using jive::util::joinNames;

  Ref<GfxNode>  world;


  viewdat.get ( world, Viewdat::WORLD_NODE );

  mrender_ = MeshRenderer::get  ( viewdat, myContext );
  sampler_ = PointSampler::find ( world,   mrender_  );

  if ( ! sampler_ )
  {
    sampler_ =

      newInstance<PointSampler> ( myName + ".pointSampler",
                                  mrender_ );

    addChild ( *world, sampler_ );
  }
  else
  {
    sampler_->sampleEvent.swap ( sampleEvent_ );
  }

  connect ( sampler_->sampleEvent,  this, & Self::pointSampled_ );
  connect ( mrender_->newMeshEvent, this, & Self::deleteMap_    );

  configure ( myProps );
  getConfig ( myConf  );
}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void PartitionView::RunData_::update

  ( const Properties&  viewdat,
    const AppContext&  apx  )

{
  using jem::toValue;
  using jem::isInstance;
  using jem::Class;
  using jem::System;
  using jem::Exception;
  using jive::fem::MeshpartModule;

  ElementSet  elems  = mrender_->getElements    ( apx );
  IdxVector   ielems = mrender_->getElemIndices ( apx );

  IdxVector   elmap;


  if ( mapFile_.size() )
  {
    elmap.resize ( elems.size() );

    print ( System::info( myName ), myName,
            " : reading block -> element map from `",
            mapFile_, "\' ...\n\n" );

    try
    {
      MeshpartModule::readElemMap ( mapFile_, elmap, elems,
                                    & System::info( myName ) );
    }
    catch ( Exception& ex )
    {
      ex.setContext ( myContext );
      throw;
    }
  }
  else
  {
    Properties   globdat = viewdat.getProps ( Viewdat::GLOBDAT );
    Ref<Object>  obj;

    if ( globdat.find( obj, fem::Globdat::ELEM_BLOCK_MAP ) )
    {
      if ( ! isInstance<IdxVector>( obj ) )
      {
        Class*  c = obj->getClass ();

        print ( System::warn(), myContext,
                " : element map is ", c->getWhatis(),
                ", but should be an index array\n" );

      }

      elmap.ref ( toValue<const IdxVector&>( obj ) );

      if ( elmap.size() != elems.size() )
      {
        print ( System::warn(), myContext,
                " : element map array has size ", elmap.size(),
                ", but should have size ", elems.size(), '\n' );
        return;
      }

      if ( jem::min( elmap ) < 0 )
      {
        print ( System::warn(), myContext,
                " : invalid element map array "
                "(negative block index)\n" );
        return;
      }
    }
  }

  if ( elmap.size() )
  {
    blockCount_ = 1 + jem::max ( elmap );

    elemBlockMap_.resize  ( ielems.size() );

    elemBlockMap_ = elmap[ielems];

    updateColors_ (        apx );
    updateInfo_   ( elmap, apx );
  }
}


//-----------------------------------------------------------------------
//   shutdown
//-----------------------------------------------------------------------


inline void PartitionView::RunData_::shutdown ()
{
  mrender_->clearColors      ();
  sampler_->sampleEvent.swap ( sampleEvent_ );
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


inline void PartitionView::RunData_::configure

  ( const Properties&  myProps )

{
  myProps.find ( mapFile_, PropNames::MAP_FILE );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


inline void PartitionView::RunData_::getConfig

  ( const Properties&  myProps ) const

{
  myProps.set ( PropNames::MAP_FILE, mapFile_ );
}


//-----------------------------------------------------------------------
//   pointSampled_
//-----------------------------------------------------------------------


void PartitionView::RunData_::pointSampled_

  ( RenderContext&  rtx,
    const Sample&   sample )

{
  using jem::Time;
  using jem::gl::LEFT_ALIGNED;
  using jem::gl::BOT_ALIGNED;
  using jem::gl::showMessageUntil;


  if ( ! blockCount_ )
  {
    sampleEvent_.emit ( rtx, sample );
  }
  else
  {
    String  text;

    idx_t   iblock;
    idx_t   ielem;
    idx_t   elemID;


    ielem = mrender_->findElement ( elemID,
                                    sample.point[0],
                                    sample.point[1],
                                    sample.point[2] );

    if ( ielem < 0 )
    {
      return;
    }

    iblock = elemBlockMap_[ielem];

    text   = String::format ( "element      : %d\n"
                              "block        : %d\n"
                              "elements     : %d\n"
                              "nodes        : %d\n"
                              "border nodes : %d",

                              elemID,
                              iblock,
                              blockInfo_[iblock].elemCount,
                              blockInfo_[iblock].nodeCount,
                              blockInfo_[iblock].bnodeCount );

    showMessageUntil ( rtx, mrender_, text,
                       Time::now() +  8_sec,
                       LEFT_ALIGNED,  BOT_ALIGNED );
  }
}


//-----------------------------------------------------------------------
//   updateColors_
//-----------------------------------------------------------------------


void PartitionView::RunData_::updateColors_

  ( const AppContext&  apx )

{
  using jem::gl::Color;
  using jem::gl::toUbyte;

  const idx_t    ielemCount = elemBlockMap_.size ();

  GLubyteMatrix  elemColors ( 4, ielemCount  );
  GLubyteMatrix  colorTable ( 3, blockCount_ );

  Color          c;


  for ( idx_t iblock = 0; iblock < blockCount_; iblock++ )
  {
    switch ( iblock )
    {
    case  0: c = Color::gray    (); break;
    case  1: c = Color::gold    (); break;
    case  2: c = Color::blue    (); break;
    case  3: c = Color::red     (); break;
    case  4: c = Color::green   (); break;
    case  5: c = Color::purple  (); break;
    case  6: c = Color::magenta (); break;
    case  7: c = Color::orange  (); break;
    case  8: c = Color::pink    (); break;
    case  9: c = Color::brown   (); break;
    case 10: c = Color::steel   (); break;
    case 11: c = Color::yellow  (); break;
    case 12: c = Color::violet  (); break;
    case 13: c = Color::silver  (); break;
    case 14: c = Color::cyan    (); break;
    case 15: c = Color::white   (); break;

    default: c = Color::random  ();
    }

    colorTable(0,iblock) = toUbyte ( c[0] );
    colorTable(1,iblock) = toUbyte ( c[1] );
    colorTable(2,iblock) = toUbyte ( c[2] );
  }

  elemColors = (GLubyte) 0x0;

  // Copy the alpha value from the mesh.

  elemColors(3,ALL) =
    toUbyte ( mrender_->getColor()[3] );

  for ( idx_t i = 0; i < ielemCount; i++ )
  {
    idx_t  iblock = elemBlockMap_[i];

    if ( iblock >= 0 && iblock < blockCount_ )
    {
      elemColors(0,i) = colorTable(0,iblock);
      elemColors(1,i) = colorTable(1,iblock);
      elemColors(2,i) = colorTable(2,iblock);
    }
  }

  mrender_->setElemColors ( elemColors, apx );
}


//-----------------------------------------------------------------------
//   updateInfo_
//-----------------------------------------------------------------------


void PartitionView::RunData_::updateInfo_

  ( const IdxVector&   elmap,
    const AppContext&  apx  )

{
  using jem::numeric::SparseUtils;
  using jive::util::Topology;

  Topology     meshTopo     =

    mrender_->getElements( apx ).toMatrix ();

  const idx_t  elemCount    = meshTopo.size (0);
  const idx_t  nodeCount    = meshTopo.size (1);

  IdxVector    elemOffsets  = meshTopo.getRowOffsets    ();
  IdxVector    nodeIndices  = meshTopo.getColumnIndices ();

  IdxVector    blockOffsets ( blockCount_ + 1 );
  IdxVector    elemIndices  ( elemCount );

  IdxMatrix    ibuf         ( nodeCount, 3 );

  IdxVector    nodeDegrees  ( ibuf[0] );
  IdxVector    nodeList     ( ibuf[1] );
  IdxVector    accu         ( ibuf[2] );

  idx_t        iblock;
  idx_t        ielem;
  idx_t        inode;

  idx_t        i, j, m, n;


  print ( jem::System::info( myName ), myName,
          " : updating partition statistics ...\n\n" );

  blockInfo_.resize ( blockCount_ );

  blockOffsets = 0;

  for ( ielem = 0; ielem < elemCount; ielem++ )
  {
    blockOffsets[elmap[ielem]]++;
  }

  SparseUtils::sumOffsets ( blockOffsets );

  for ( ielem = 0; ielem < elemCount; ielem++ )
  {
    iblock = elmap[ielem];
    elemIndices[blockOffsets[iblock]++] = ielem;
  }

  SparseUtils::shiftOffsets ( blockOffsets );

  nodeDegrees = 0;
  accu        = 0;

  n = nodeIndices.size ();

  for ( i = 0; i < n; i++ )
  {
    nodeDegrees[nodeIndices[i]]++;
  }

  for ( iblock = 0; iblock < blockCount_; iblock++ )
  {
    BlockInfo&  binfo = blockInfo_[iblock];

    i                 = blockOffsets[iblock];
    m                 = blockOffsets[iblock + 1];

    binfo.elemCount   = m - i;
    binfo.nodeCount   = 0;
    binfo.bnodeCount  = 0;

    for ( ; i < m; i++ )
    {
      ielem = elemIndices[i];
      j     = elemOffsets[ielem];
      n     = elemOffsets[ielem + 1];

      for ( ; j < n; j++ )
      {
        inode = nodeIndices[j];

        if ( accu[inode] == 0 )
        {
          nodeList[binfo.nodeCount++] = inode;
        }

        accu[inode]++;
      }
    }

    for ( i = 0; i < binfo.nodeCount; i++ )
    {
      inode = nodeList[i];

      if ( accu[inode] < nodeDegrees[inode] )
      {
        binfo.bnodeCount++;
      }

      accu[inode] = 0;
    }
  }
}


//-----------------------------------------------------------------------
//   deleteMap_
//-----------------------------------------------------------------------


void PartitionView::RunData_::deleteMap_ ()
{
  if ( blockCount_ )
  {
    blockCount_ = 0;

    blockInfo_   .resize ( 0 );
    elemBlockMap_.resize ( 0 );
  }
}


//=======================================================================
//   class PartitionView
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  PartitionView::TYPE_NAME = "PartitionView";


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


PartitionView::PartitionView ( const String& name ) :

  Super ( name )

{}


PartitionView::~PartitionView ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


void PartitionView::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  viewdat,
    const AppContext&  apx  )

{
  rundat_ = nullptr;

  Properties     myConf  = conf .makeProps ( myName_ );
  Properties     myProps = props.getProps  ( myName_ );

  Ref<RunData_>  newdat  =

    newInstance<RunData_> ( myName_, getContext() );

  newdat->init ( myConf, myProps, viewdat );
  rundat_.swap ( newdat );
}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void PartitionView::update

  ( const Properties&  viewdat,
    const AppContext&  apx )

{
  if ( rundat_ )
  {
    rundat_->update ( viewdat, apx );
  }
}


//-----------------------------------------------------------------------
//   shutdown
//-----------------------------------------------------------------------


void PartitionView::shutdown

  ( const Properties&  viewdat,
    const AppContext&  apx )

{
  if ( rundat_ )
  {
    rundat_->shutdown ();
    rundat_ = nullptr;
  }
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void PartitionView::configure

  ( const Properties&  props,
    const Properties&  viewdat,
    const AppContext&  apx )

{
  if ( rundat_ && props.contains( myName_ ) )
  {
    rundat_->configure ( props.findProps( myName_ ) );
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void PartitionView::getConfig ( const Properties& props ) const
{
  if ( rundat_ )
  {
    rundat_->getConfig ( props.makeProps( myName_ ) );
  }
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void PartitionView::declare ()
{
  PluginFactory::declare ( TYPE_NAME,  & makeNew );
  PluginFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Plugin> PartitionView::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props )

{
  return newInstance<Self> ( name );
}


JIVE_END_PACKAGE( gl )
