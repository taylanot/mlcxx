
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


#include <cmath>
#include <jem/base/assert.h>
#include <jem/base/Time.h>
#include <jem/base/System.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalInputException.h>
#include <jem/gl/input.h>
#include <jem/gl/utilities.h>
#include <jem/gl/Compound.h>
#include <jem/gl/EventNode.h>
#include <jem/gl/GroupNode.h>
#include <jem/gl/MessageBox.h>
#include <jem/gl/PointSampler.h>
#include <jem/gl/RenderContext.h>
#include <jem/io/list.h>
#include <jem/io/NumberFormatter.h>
#include <jem/util/None.h>
#include <jem/util/Event.h>
#include <jem/util/Properties.h>
#include <jive/util/utilities.h>
#include <jive/util/Assignable.h>
#include <jive/geom/ShapeTable.h>
#include <jive/fem/ElementGroup.h>
#include <jive/gl/declare.h>
#include <jive/gl/Names.h>
#include <jive/gl/Viewdat.h>
#include <jive/gl/StdGfxMesher.h>
#include <jive/gl/MeshRenderer.h>
#include <jive/gl/VectorField.h>
#include <jive/gl/ChainPlugin.h>
#include <jive/gl/UserconfPlugin.h>
#include <jive/gl/PluginFactory.h>
#include <jive/gl/MeshView.h>


JEM_DEFINE_CLASS( jive::gl::MeshView );


JIVE_BEGIN_PACKAGE( gl )


using jem::newInstance;
using jem::io::PrintWriter;
using jem::gl::EventNode;
using jem::gl::GfxNode;
using jem::gl::GroupNode;
using jem::gl::PointSampler;


//=======================================================================
//   class MeshView::RunData_
//=======================================================================


class MeshView::RunData_ : public jem::Collectable
{
 public:

  typedef RunData_          Self;
  typedef
    PointSampler::Sample    Sample;


  explicit inline           RunData_

    ( const String&           ctx );

  void                      init

    ( const String&           myName,
      const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat,
      const AppContext&       apx );

  void                      update

    ( const Properties&       globdat,
      const AppContext&       apx );

  void                      configure

    ( const Properties&       myProps,
      const Properties&       viewdat,
      const AppContext&       apx );

  inline void               getConfig

    ( const Properties&       myProps )      const;


 public:

  const String              myContext;

  Ref<GroupNode>            mgroup;
  Ref<EventNode>            evNode;
  Ref<MeshRenderer>         mrender;


 private:

  void                      initRenderer_

    ( const String&           myName,
      const ElementGroup&     elems,
      const Properties&       globdat,
      const AppContext&       apx );

  void                      getMeshSize_

    ( const AppContext&       apx );

  void                      pointSampled_

    ( RenderContext&          rtx,
      const Sample&           sample );

  void                      geomChanged_  ();


 private:

  Ref<VectorField>          deform_;

  double                    meshSize_;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline MeshView::RunData_::RunData_ ( const String& ctx ) :

  myContext ( ctx )

{
  meshSize_ = -1.0;
}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


void MeshView::RunData_::init

  ( const String&      myName,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  viewdat,
    const AppContext&  apx )

{
  using jem::gl::addChild;
  using jem::util::connect;
  using jem::util::None;


  Properties  myConf  = conf .makeProps   ( myName );
  Properties  myProps = props.findProps   ( myName );

  Properties  config0 = viewdat.makeProps ( Viewdat::CONFIG0 );
  Properties  globdat = viewdat.getProps  ( Viewdat::GLOBDAT );

  Ref<PointSampler>  psamp;
  Ref<GfxNode>       world;


  if ( ! myProps.contains( PropNames::ELEMENTS ) )
  {
    myProps.set ( PropNames::ELEMENTS, "all" );
  }

  viewdat.get   ( world, Viewdat::WORLD_NODE );

  initRenderer_ (
    myName,
    ElementGroup::get ( myConf, myProps, globdat, myContext ),
    globdat,
    apx
  );

  mgroup = newInstance<GroupNode> ( myName );
  evNode = newInstance<EventNode> ( myName + ".events" );

  mgroup->addChild  ( evNode  );
  mgroup->addChild  ( mrender );

  addChild  ( *world, mgroup  );

  mgroup->configure ( props   );
  mgroup->getConfig ( conf    );
  mgroup->getConfig ( config0 );

  configure ( myProps, viewdat, apx );
  getConfig ( myConf );

  psamp = newInstance<PointSampler> ( myName + ".pointSampler",
                                      mrender );

  mgroup->addChild ( psamp );

  connect ( psamp->sampleEvent, this, & Self::pointSampled_ );
}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void MeshView::RunData_::update

  ( const Properties&  globdat,
    const AppContext&  apx )

{
  using jem::System;
  using jem::IllegalInputException;

  try
  {
    mrender->update ( apx );
  }
  catch ( IllegalInputException& ex )
  {
    ex.setContext ( myContext );
    throw;
  }

  if ( deform_ )
  {
    Matrix  delta ( 3, mrender->getNodes(apx).size() );

    double  scale;


    if ( deform_->autoScale )
    {
      if ( meshSize_ < 0.0 )
      {
        getMeshSize_ ( apx );
      }

      deform_->setPostScale ( meshSize_ );
    }

    try
    {
      deform_->getData  ( scale, delta, globdat, myContext, apx );
      mrender->addToNodeCoords ( delta, apx );
    }
    catch ( const jem::IllegalInputException& ex )
    {
      print ( System::warn(), ex, '\n' );
    }
  }
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void MeshView::RunData_::configure

  ( const Properties&  myProps,
    const Properties&  viewdat,
    const AppContext&  apx )

{
  using jem::util::isNone;
  using jem::util::connect;
  using jem::util::disconnect;


  JEM_PRECHECK ( mrender );

  NodeSet      nodes = mrender->getNodes ( apx );

  Ref<Object>  obj;


  if ( ! myProps.find( obj, PropNames::DEFORMATION ) )
  {
    return;
  }

  if ( isNone( obj ) && deform_ )
  {
    disconnect ( nodes.newGeomEvent(), this, & Self::geomChanged_ );
    disconnect ( nodes.newSizeEvent(), this, & Self::geomChanged_ );

    mrender->trackMeshGeom ();

    deform_ = nullptr;

    return;
  }

  if ( ! deform_ )
  {
    String   compNames[3] = { "dx", "dy", "dz" };

    deform_ = newInstance<VectorField> ( nodes.rank    (),
                                         nodes.getData (),
                                         compNames );

    connect ( nodes.newGeomEvent(), this, & Self::geomChanged_ );
    connect ( nodes.newSizeEvent(), this, & Self::geomChanged_ );

    meshSize_ = -1.0;
  }

  Properties  dataSets = viewdat.findProps ( Viewdat::DATA_SETS );

  deform_->configure ( PropNames::DEFORMATION, myProps, dataSets );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


inline void MeshView::RunData_::getConfig

  ( const Properties&  myProps ) const

{
  using jem::util::None;

  if ( deform_ )
  {
    deform_->getConfig

      ( myProps.makeProps( PropNames::DEFORMATION ) );
  }
  else
  {
    myProps.set ( PropNames::DEFORMATION, None::getInstance() );
  }
}


//-----------------------------------------------------------------------
//   initRenderer_
//-----------------------------------------------------------------------


void MeshView::RunData_::initRenderer_

  ( const String&        myName,
    const ElementGroup&  egroup,
    const Properties&    globdat,
    const AppContext&    apx )

{
  using jive::geom::ShapeTable;

  ElementSet        elems  = egroup.getElements ();
  NodeSet           nodes  = elems .getNodes    ();

  Ref<ShapeTable>   shapes =

    ShapeTable::get ( elems.getData(), globdat, myContext );

  Ref<GfxMesher>    mesher =

    newInstance<StdGfxMesher> ();


  mrender = newInstance<MeshRenderer> ( myName, apx,
                                        egroup, shapes, mesher );
}


//-----------------------------------------------------------------------
//   getMeshSize_
//-----------------------------------------------------------------------


void MeshView::RunData_::getMeshSize_ ( const AppContext& apx )
{
  NodeSet      nodes  = mrender->getNodes ( apx );
  IdxVector    inodes = mrender->getVxNodeMap  ();

  const idx_t  m      = nodes .rank ();
  const idx_t  n      = inodes.size ();

  Matrix       coords = nodes.toMatrix ();

  Vector       xmin   ( m );
  Vector       xmax   ( m );

  double       x;

  idx_t        i, j, k;


  for ( j = 0; j < n; j++ )
  {
    if ( inodes[j] >= 0 )
    {
      break;
    }
  }

  if ( j >= n )
  {
    meshSize_ = 1.0;
    return;
  }

  k = inodes[j];

  for ( i = 0; i < m; i++ )
  {
    xmin[i] = xmax[i] = coords(i,k);
  }

  for ( ; j < n; j++ )
  {
    k = inodes[j];

    if ( k < 0 )
    {
      continue;
    }

    for ( i = 0; i < m; i++ )
    {
      x = coords(i,k);

      if      ( x < xmin[i] )
      {
        xmin[i] = x;
      }
      else if ( x > xmax[i] )
      {
        xmax[i] = x;
      }
    }
  }

  meshSize_ = 0.0;

  for ( i = 0; i < m; i++ )
  {
    x = xmax[i] - xmin[i];

    if ( x > meshSize_ )
    {
      meshSize_ = x;
    }
  }

  if ( meshSize_ <= 0.0 )
  {
    meshSize_ = 1.0;
  }
}


//-----------------------------------------------------------------------
//   pointSampled_
//-----------------------------------------------------------------------


void MeshView::RunData_::pointSampled_

  ( RenderContext&  rtx,
    const Sample&   sample )

{
  using jem::Time;
  using jem::gl::LEFT_ALIGNED;
  using jem::gl::BOT_ALIGNED;
  using jem::gl::clamped;
  using jem::gl::showMessageUntil;

  double     eps;
  int        prec[3];
  idx_t      elemID;
  idx_t      ielem;


  ielem = mrender->findElement ( elemID,
                                 sample.point[0],
                                 sample.point[1],
                                 sample.point[2] );

  if ( ielem < 0 )
  {
    return;
  }

  // Determine the number of digits to show.

  for ( idx_t i = 0; i < 3; i++ )
  {
    if ( std::fabs( sample.point[i] ) > 0.0 )
    {
      eps = sample.error[i] / std::fabs ( sample.point[i] );
    }
    else
    {
      eps = 1.0;
    }

    if ( eps > 0.0 )
    {
      prec[i] = (int) std::ceil ( -std::log10 ( eps ) );
      prec[i] = clamped ( prec[i], 1, 6 );
    }
    else
    {
      prec[i] = 6;
    }
  }

  String  text =

    String::format ( "element : % d\n"
                     "x-coord : % .*g\n"
                     "y-coord : % .*g\n"
                     "z-coord : % .*g",

                     elemID,
                     prec[0], sample.point[0],
                     prec[1], sample.point[1],
                     prec[2], sample.point[2] );

  showMessageUntil ( rtx, mgroup,  text,
                     Time::now() + 10_sec,
                     LEFT_ALIGNED, BOT_ALIGNED );
}


//-----------------------------------------------------------------------
//   geomChanged_
//-----------------------------------------------------------------------


void MeshView::RunData_::geomChanged_ ()
{
  meshSize_ = -1.0;
}


//=======================================================================
//   class MeshView::Scope_
//=======================================================================


class MeshView::Scope_
{
 public:

  typedef MeshRenderer      MRenderer;

  inline                    Scope_

    ( const Properties&       viewdat,
      const RunData_&         rundat );


  inline                   ~Scope_ ();


 public:

  Ref<GfxNode>              world;
  Ref<EventNode>            evNode;


 private:

  const Properties&         viewdat_;

};


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


inline MeshView::Scope_::Scope_

  ( const Properties&      viewdat,
    const RunData_&        rundat ) :

    viewdat_ ( viewdat )

{
  viewdat_.get ( world,  Viewdat::WORLD_NODE );
  viewdat_.get ( evNode, Viewdat::EVENT_NODE );

  viewdat_.set ( Viewdat::WORLD_NODE,    rundat.mgroup  );
  viewdat_.set ( Viewdat::EVENT_NODE,    rundat.evNode  );
  viewdat_.set ( Viewdat::MESH_RENDERER, rundat.mrender );
}


inline MeshView::Scope_::~Scope_ ()
{
  viewdat_.erase ( Viewdat::MESH_RENDERER );

  viewdat_.set   ( Viewdat::WORLD_NODE, world  );
  viewdat_.set   ( Viewdat::EVENT_NODE, evNode );
}


//=======================================================================
//   class MeshView
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  MeshView::TYPE_NAME = "MeshView";


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


MeshView::MeshView ( const String& name ) :

  Super ( name )

{
  plugins_ = newInstance<ChainPlugin> ( myName_ );
}


MeshView::~MeshView ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


void MeshView::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  viewdat,
    const AppContext&  apx )

{
  rundat_ = nullptr;

  Ref<RunData_>  newdat = newInstance<RunData_> ( getContext() );

  newdat->init ( myName_, conf, props, viewdat, apx );

  {
    Scope_  scope ( viewdat, *newdat );

    plugins_->init ( conf, props, viewdat, apx );
  }

  // Init succeeded, so switch to running mode

  rundat_.swap ( newdat );
}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void MeshView::update

  ( const Properties&  viewdat,
    const AppContext&  apx )

{
  if ( rundat_ )
  {
    RunData_&  d = * rundat_;

    d.update ( viewdat.getProps( Viewdat::GLOBDAT ), apx );

    {
      Scope_  scope ( viewdat, d );

      plugins_->update ( viewdat, apx  );
    }
  }
}


//-----------------------------------------------------------------------
//   shutdown
//-----------------------------------------------------------------------


void MeshView::shutdown

  ( const Properties&  viewdat,
    const AppContext&  apx )

{
  using jem::gl::toCompound;

  if ( rundat_ )
  {
    RunData_&  d = * rundat_;

    Scope_     scope ( viewdat, d );

    plugins_->shutdown ( viewdat, apx );

    toCompound(*scope.world).eraseChild ( d.mgroup );
  }

  rundat_ = nullptr;
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void MeshView::configure

  ( const Properties&  props,
    const Properties&  viewdat,
    const AppContext&  apx )

{
  if ( ! rundat_ )
  {
    plugins_->configure ( props, viewdat, apx );
  }
  else
  {
    RunData_&  d = * rundat_;

    if ( props.contains( myName_ ) )
    {
      d.configure ( props.findProps( myName_ ), viewdat, apx );
    }

    {
      Scope_  scope ( viewdat, d );

      plugins_->configure ( props, viewdat, apx );
    }
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void MeshView::getConfig ( const Properties& props ) const
{
  plugins_->getConfig ( props );

  if ( rundat_ )
  {
    rundat_->getConfig ( props.makeProps( myName_ ) );
  }
}


//-----------------------------------------------------------------------
//   printHelp
//-----------------------------------------------------------------------


void MeshView::printHelp ( PrintWriter& pr ) const
{
  using jem::io::blankLine;
  using jem::io::beginItem;
  using jem::io::endItem;

  if ( rundat_ )
  {
    print ( pr, blankLine );
    print ( pr, "--- Controls for object `",
            rundat_->mrender->getName(), "\' ---" );
    print ( pr, blankLine );

    print ( pr, beginItem( "Left mouse click" ) );
    print ( pr,   "Show information about the element " );
    print ( pr,   "under the mouse cursor." );
    print ( pr, endItem );

    plugins_->printHelp ( pr );
  }
}


//-----------------------------------------------------------------------
//   addUserPlugin
//-----------------------------------------------------------------------


void MeshView::addUserPlugin ()
{
  plugins_->pushBack ( newInstance<UserconfPlugin>( myName_ ) );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void MeshView::declare ()
{
  PluginFactory::declare ( TYPE_NAME,  & makeNew );
  PluginFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Plugin> MeshView::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props )

{
  Ref<Self>  self = newInstance<Self> ( name );

  self->addUserPlugin ();

  return self;
}


JIVE_END_PACKAGE( gl )
