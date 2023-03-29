
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
#include <jem/gl/input.h>
#include <jem/gl/Compound.h>
#include <jem/gl/VertexSet.h>
#include <jem/gl/MarkerSet.h>
#include <jem/util/Event.h>
#include <jem/util/Properties.h>
#include <jive/util/utilities.h>
#include <jive/util/ItemMap.h>
#include <jive/util/ItemGroup.h>
#include <jive/fem/NodeSet.h>
#include <jive/gl/Names.h>
#include <jive/gl/AppRef.h>
#include <jive/gl/Viewdat.h>
#include <jive/gl/MeshRenderer.h>
#include <jive/gl/PluginFactory.h>
#include <jive/gl/MarkerStyleTable.h>
#include <jive/gl/MeshNodeView.h>


JEM_DEFINE_CLASS( jive::gl::MeshNodeView );


JIVE_BEGIN_PACKAGE( gl )


using jem::newInstance;
using jem::Array;
using jem::io::PrintWriter;
using jem::gl::MarkerSet;


//=======================================================================
//   class MeshNodeView::RunData_
//=======================================================================


class MeshNodeView::RunData_ : public jem::Collectable
{
 public:

  typedef RunData_          Self;


  explicit inline           RunData_

    ( const String&           ctx );

  void                      init

    ( const String&           myName,
      const Properties&       props,
      const Properties&       conf,
      const Properties&       viewdat,
      const AppContext&       apx );

  void                      update

    ( const Properties&       viewdat,
      const AppContext&       apx );

  inline void               shutdown        ();

  inline void               configure

    ( const Properties&       myProps,
      const AppContext&       apx );

  void                      getConfig

    ( const Properties&       myProps )        const;


 public:

  const String              myContext;

  bool                      updated;


 private:

  void                      markOutdated_   ();

  void                      selectNodes_

    ( const StringVector&     specs,
      const AppContext&       apx );

  void                      updateGroup_

    ( idx_t                   i,
      const Ref<ItemGroup>&   group,
      const AppContext&       apx );


 private:

  Ref<MeshRenderer>         mrender_;
  Ref<MarkerSet>            markers_;
  Ref<MarkerStyleTable>     styles_;

  StringVector              groupNames_;
  IdxVector                 groupStyles_;

  Array <
    AppRef<ItemGroup> >     groups_;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline MeshNodeView::RunData_::RunData_ ( const String& ctx ) :

  myContext ( ctx )

{
  updated = true;
}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


void MeshNodeView::RunData_::init

  ( const String&      myName,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  viewdat,
    const AppContext&  apx )

{
  using jem::gl::addChild;
  using jem::gl::GfxNode;
  using jem::gl::VertexSet;
  using jem::util::connect;

  Properties      myConf  = conf   .makeProps ( myName );
  Properties      myProps = props  .findProps ( myName );

  Properties      config0 = viewdat.makeProps ( Viewdat::CONFIG0 );

  Ref<GfxNode>    world;
  Ref<VertexSet>  vertices;


  viewdat.get ( world, Viewdat::WORLD_NODE );

  mrender_ = MeshRenderer::get      ( viewdat, myContext );
  vertices = mrender_->getVertices  ();
  markers_ = newInstance<MarkerSet> ( myName, vertices );
  styles_  = MarkerStyleTable::get  ( viewdat );

  markers_->configure ( props   );
  markers_->getConfig ( conf    );
  markers_->getConfig ( config0 );

  if ( ! myProps.contains( PropNames::NODES ) )
  {
    myProps.set ( PropNames::NODES, "all" );
  }

  configure ( myProps, apx );
  getConfig ( myConf );

  addChild  ( *world, markers_ );

  if ( ! updated )
  {
    update ( viewdat, apx );
  }

  connect ( vertices->newShapeEvent, this, & Self::markOutdated_ );
}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void MeshNodeView::RunData_::update

  ( const Properties&  viewdat,
    const AppContext&  apx )

{
  using jem::System;
  using jive::util::ItemMap;

  Properties      globdat   = viewdat.getProps ( Viewdat::GLOBDAT );

  NodeSet         nodes     = mrender_->getNodes     ( apx );
  IdxVector       nodeVxMap = mrender_->getNodeVxMap ( apx );
  Ref<ItemMap>    imap      = nodes.getData()->getItemMap ();

  Ref<ItemGroup>  group;

  IdxVector       nodeIDs;
  IdxVector       inodes;

  idx_t           ivx;
  idx_t           i, j, k, n;


  markers_->unmarkAll ();

  for ( i = 0; i < groups_.size(); i++ )
  {
    group = ItemGroup::find ( groupNames_[i],
                              nodes.getData(), globdat );

    if ( groups_[i][apx] != group )
    {
      updateGroup_ ( i, group, apx );
    }

    if ( ! group )
    {
      print ( System::warn(), myContext,
              " : undefined node group: ", groupNames_[i],
              " (skipping)\n" );

      continue;
    }

    group->resetEvents ();

    n = group->size ();

    inodes .resize ( n );
    nodeIDs.resize ( n );

    inodes = group->getIndices ();

    imap->getItemIDs ( nodeIDs, inodes );

    k = 0;

    for ( j = 0; j < n; j++ )
    {
      ivx = nodeVxMap[inodes[j]];

      if ( ivx >= 0 )
      {
        inodes[k]  = ivx;
        nodeIDs[k] = nodeIDs[j];
        k++;
      }
    }

    markers_->markVertices ( inodes[slice(BEGIN,k)],
                             groupStyles_[i],
                             nodeIDs[slice(BEGIN,k)] );
  }

  updated = true;
}


//-----------------------------------------------------------------------
//   shutdown
//-----------------------------------------------------------------------


inline void MeshNodeView::RunData_::shutdown ()
{
  jem::gl::eraseFromParent ( markers_ );
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void MeshNodeView::RunData_::configure

  ( const Properties&  myProps,
    const AppContext&  apx )

{
  StringVector  specs;

  if ( myProps.find( specs, PropNames::NODES ) )
  {
    try
    {
      selectNodes_ ( specs, apx );
    }
    catch ( jem::IllegalInputException& ex )
    {
      ex.setContext ( myProps.getContext( PropNames::NODES ) );
      throw;
    }
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void MeshNodeView::RunData_::getConfig

  ( const Properties&  myProps ) const

{
  const idx_t   n = groupNames_.size ();

  StringVector  specs ( n );


  for ( idx_t i = 0; i < n; i++ )
  {
    specs[i] = groupNames_[i] + " : " +
               markers_->getStyleName ( groupStyles_[i] );
  }

  myProps.set ( PropNames::NODES, specs );
}



//-----------------------------------------------------------------------
//   markOutdated_
//-----------------------------------------------------------------------


void MeshNodeView::RunData_::markOutdated_ ()
{
  updated = false;
}


//-----------------------------------------------------------------------
//   selectNodes_
//-----------------------------------------------------------------------


void MeshNodeView::RunData_::selectNodes_

  ( const StringVector&  specs,
    const AppContext&    apx )

{
  using jem::util::disconnect;

  const idx_t   n = specs.size ();

  StringVector  newNames  ( n );
  IdxVector     newStyles ( n );

  String        styleName;
  String        s;

  idx_t         istyle;
  idx_t         i, j;


  for ( i = 0; i < n; i++ )
  {
    s = specs[i];
    j = s.rfind ( ':' );

    if ( j < 0 )
    {
      newNames[i] = s.stripWhite ();
      styleName   = "numbers";
    }
    else
    {
      newNames[i] = s[slice(BEGIN,j)]  .stripWhite ();
      styleName   = s[slice(j + 1,END)].stripWhite ();
    }

    if ( ! newNames[i].size() )
    {
      throw jem::IllegalInputException (
        CLASS_NAME,
        "empty node group name"
      );
    }

    if ( ! styleName.size() )
    {
      throw jem::IllegalInputException (
        CLASS_NAME,
        "empty marker style name"
      );
    }

    istyle = markers_->addStyle (
      styleName,
      styles_->getStyle ( styleName, myContext )
    );

    newStyles[i] = istyle;
  }

  for ( i = 0; i < groups_.size(); i++ )
  {
    if ( groups_[i][apx] )
    {
      disconnect ( groups_[i][apx]->newSizeEvent,
                   this, & Self::markOutdated_ );

      groups_[i][apx] = nullptr;
    }
  }

  groupNames_ .swap ( newNames  );
  groupStyles_.swap ( newStyles );

  groups_.reshape   ( n );

  updated = false;
}


//-----------------------------------------------------------------------
//   updateGroup_
//-----------------------------------------------------------------------


void MeshNodeView::RunData_::updateGroup_

  ( idx_t                    i,
    const Ref<ItemGroup>&  group,
    const AppContext&      apx )

{
  using jem::util::connect;
  using jem::util::disconnect;

  if ( groups_[i][apx] )
  {
    disconnect ( groups_[i][apx]->newSizeEvent,
                 this, & Self::markOutdated_ );
  }

  groups_[i][apx] = group;

  if ( group )
  {
    connect    ( group->newSizeEvent,
                 this, & Self::markOutdated_ );
  }
}


//=======================================================================
//   class MeshNodeView
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  MeshNodeView::TYPE_NAME = "MeshNodeView";


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


MeshNodeView::MeshNodeView ( const String& name ) :

  Super ( name )

{}


MeshNodeView::~MeshNodeView ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


void MeshNodeView::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  viewdat,
    const AppContext&  apx )

{
  rundat_ = nullptr;

  Ref<RunData_>  newdat = newInstance<RunData_> ( getContext() );

  newdat->init ( myName_, conf, props, viewdat, apx );
  rundat_.swap ( newdat );
}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void MeshNodeView::update

  ( const Properties&  viewdat,
    const AppContext&  apx )

{
  if ( rundat_ && ! rundat_->updated )
  {
    rundat_->update ( viewdat, apx );
  }
}


//-----------------------------------------------------------------------
//   shutdown
//-----------------------------------------------------------------------


void MeshNodeView::shutdown

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


void MeshNodeView::configure

  ( const Properties&  props,
    const Properties&  viewdat,
    const AppContext&  apx )

{
  if ( rundat_ && props.contains( myName_ ) )
  {
    rundat_->configure ( props.findProps( myName_ ), apx );
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void MeshNodeView::getConfig ( const Properties& props ) const
{
  if ( rundat_ )
  {
    rundat_->getConfig ( props.makeProps( myName_ ) );
  }
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void MeshNodeView::declare ()
{
  PluginFactory::declare ( TYPE_NAME,  & makeNew );
  PluginFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Plugin> MeshNodeView::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props )

{
  return newInstance<Self> ( name );
}


JIVE_END_PACKAGE( gl )
