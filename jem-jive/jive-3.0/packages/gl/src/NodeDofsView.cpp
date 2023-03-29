
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
#include <jive/util/DofSpace.h>
#include <jive/fem/NodeSet.h>
#include <jive/gl/Names.h>
#include <jive/gl/AppRef.h>
#include <jive/gl/Viewdat.h>
#include <jive/gl/MeshRenderer.h>
#include <jive/gl/PluginFactory.h>
#include <jive/gl/MarkerStyleTable.h>
#include <jive/gl/NodeDofsView.h>


JEM_DEFINE_CLASS( jive::gl::NodeDofsView );


JIVE_BEGIN_PACKAGE( gl )


using jem::newInstance;
using jem::Array;
using jem::IllegalInputException;
using jem::io::PrintWriter;
using jem::gl::MarkerSet;


//=======================================================================
//   class NodeDofsView::RunData_
//=======================================================================


class NodeDofsView::RunData_ : public jem::Collectable
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

    ( const Properties&       myProps );

  void                      getConfig

    ( const Properties&       myProps )        const;


 public:

  const String              myContext;

  bool                      updated;


 private:

  void                      markOutdated_   ();

  void                      selectDofs_

    ( const StringVector&     specs );


 private:

  Ref<MeshRenderer>         mrender_;
  Ref<MarkerSet>            markers_;
  Ref<MarkerStyleTable>     styles_;

  AppRef<DofSpace>          dofs_;
  StringVector              dofTypes_;
  IdxVector                 dofStyles_;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline NodeDofsView::RunData_::RunData_ ( const String& ctx ) :

  myContext ( ctx )

{
  updated = true;
}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


void NodeDofsView::RunData_::init

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
  Properties      globdat = viewdat.getProps  ( Viewdat::GLOBDAT );

  Ref<GfxNode>    world;
  Ref<VertexSet>  vertices;


  viewdat.get ( world,  Viewdat::WORLD_NODE );

  mrender_ = MeshRenderer::get      ( viewdat, myContext );
  vertices = mrender_->getVertices  ();
  markers_ = newInstance<MarkerSet> ( myName, vertices );
  styles_  = MarkerStyleTable::get  ( viewdat );

  dofs_    = DofSpace::get ( mrender_->getNodes(apx).getData(),
                             globdat, myContext );

  markers_->configure ( props   );
  markers_->getConfig ( conf    );
  markers_->getConfig ( config0 );

  if ( ! myProps.contains( PropNames::DOFS ) )
  {
    myProps.set ( PropNames::DOFS, "all" );
  }

  configure ( myProps );
  getConfig ( myConf  );

  addChild  ( *world, markers_ );

  if ( ! updated )
  {
    update ( viewdat, apx );
  }

  connect ( vertices  ->newShapeEvent, this, & Self::markOutdated_ );
  connect ( dofs_[apx]->newSizeEvent,  this, & Self::markOutdated_ );
  connect ( dofs_[apx]->newOrderEvent, this, & Self::markOutdated_ );
}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void NodeDofsView::RunData_::update

  ( const Properties&  viewdat,
    const AppContext&  apx )

{
  using jem::System;

  IdxVector    nodeVxMap = mrender_->getNodeVxMap ( apx );

  const idx_t  nodeCount = nodeVxMap.size ();

  IdxMatrix    ibuf      ( nodeCount, 2 );

  IdxVector    idofs     ( ibuf[0] );
  IdxVector    inodes    ( ibuf[1] );

  idx_t        itype;
  idx_t        ivx;
  idx_t        i, j, k, n;


  markers_->unmarkAll ();

  for ( i = 0; i < dofTypes_.size(); i++ )
  {
    itype = dofs_[apx]->findType ( dofTypes_[i] );

    if ( itype < 0 )
    {
      print ( System::warn(), myContext,
              " : undefined DOF type: `",
              dofTypes_[i], "\' (skipping)\n" );

      continue;
    }

    n = dofs_[apx]->getDofsForType ( idofs, inodes, itype );

    k = 0;

    for ( j = 0; j < n; j++ )
    {
      ivx = nodeVxMap[inodes[j]];

      if ( ivx >= 0 )
      {
        inodes[k] = ivx;
        idofs [k] = idofs[j];
        k++;
      }
    }

    markers_->markVertices ( inodes[slice(BEGIN,k)],
                             dofStyles_[i],
                             idofs[slice(BEGIN,k)] );
  }

  dofs_[apx]->resetEvents  ();

  updated = true;
}


//-----------------------------------------------------------------------
//   shutdown
//-----------------------------------------------------------------------


inline void NodeDofsView::RunData_::shutdown ()
{
  jem::gl::eraseFromParent ( markers_ );
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void NodeDofsView::RunData_::configure

  ( const Properties&  myProps )

{
  StringVector  specs;

  if ( myProps.find( specs, PropNames::DOFS ) )
  {
    try
    {
      selectDofs_ ( specs );
    }
    catch ( IllegalInputException& ex )
    {
      ex.setContext ( myProps.getContext( PropNames::DOFS ) );
      throw;
    }
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void NodeDofsView::RunData_::getConfig

  ( const Properties&  myProps ) const

{
  const idx_t   n = dofTypes_.size ();

  StringVector  specs ( n );


  for ( idx_t i = 0; i < n; i++ )
  {
    specs[i] = dofTypes_[i] + " : " +
               markers_->getStyleName ( dofStyles_[i] );
  }

  myProps.set ( PropNames::DOFS, specs );
}


//-----------------------------------------------------------------------
//   markOutdated_
//-----------------------------------------------------------------------


void NodeDofsView::RunData_::markOutdated_ ()
{
  updated = false;
}


//-----------------------------------------------------------------------
//   selectDofs_
//-----------------------------------------------------------------------


void NodeDofsView::RunData_::selectDofs_

  ( const StringVector&  specs )

{
  const idx_t   n = specs.size ();

  StringVector  newTypes  ( n );
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
      newTypes[i] = s.stripWhite ();
      styleName   = "numbers";
    }
    else
    {
      newTypes[i] = s[slice(BEGIN,j)]  .stripWhite ();
      styleName   = s[slice(j + 1,END)].stripWhite ();
    }

    if ( ! newTypes[i].size() )
    {
      throw IllegalInputException (
        myContext,
        "empty DOF type name"
      );
    }

    if ( ! styleName.size() )
    {
      throw IllegalInputException (
        myContext,
        "empty marker style name"
      );
    }

    istyle = markers_->addStyle (
      styleName,
      styles_->getStyle ( styleName, myContext )
    );

    newStyles[i] = istyle;
  }

  dofTypes_ .swap ( newTypes  );
  dofStyles_.swap ( newStyles );

  updated = false;
}


//=======================================================================
//   class NodeDofsView
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  NodeDofsView::TYPE_NAME = "NodeDofsView";


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


NodeDofsView::NodeDofsView ( const String& name ) :

  Super ( name )

{}


NodeDofsView::~NodeDofsView ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


void NodeDofsView::init

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


void NodeDofsView::update

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


void NodeDofsView::shutdown

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


void NodeDofsView::configure

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


void NodeDofsView::getConfig ( const Properties& props ) const
{
  if ( rundat_ )
  {
    rundat_->getConfig ( props.makeProps( myName_ ) );
  }
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void NodeDofsView::declare ()
{
  PluginFactory::declare ( TYPE_NAME,  & makeNew );
  PluginFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Plugin> NodeDofsView::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props )

{
  return newInstance<Self> ( name );
}


JIVE_END_PACKAGE( gl )
