
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


#include <jem/base/Time.h>
#include <jem/base/System.h>
#include <jem/base/StringBuffer.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalInputException.h>
#include <jem/base/array/select.h>
#include <jem/base/array/utilities.h>
#include <jem/gl/GroupNode.h>
#include <jem/gl/VertexSet.h>
#include <jem/gl/MarkerSet.h>
#include <jem/gl/MessageBox.h>
#include <jem/gl/PointSampler.h>
#include <jem/gl/RenderContext.h>
#include <jem/util/None.h>
#include <jem/util/Event.h>
#include <jem/util/Properties.h>
#include <jive/util/utilities.h>
#include <jive/util/ItemMap.h>
#include <jive/util/ItemGroup.h>
#include <jive/util/PointSet.h>
#include <jive/geom/boxes.h>
#include <jive/gl/Names.h>
#include <jive/gl/AppRef.h>
#include <jive/gl/GLArray.h>
#include <jive/gl/Viewdat.h>
#include <jive/gl/VectorField.h>
#include <jive/gl/PluginFactory.h>
#include <jive/gl/MarkerStyleTable.h>
#include <jive/gl/PointView.h>


JEM_DEFINE_CLASS( jive::gl::PointView );


JIVE_BEGIN_PACKAGE( gl )


using jem::newInstance;
using jem::Array;
using jem::IllegalInputException;
using jem::gl::GroupNode;
using jem::gl::VertexSet;
using jem::gl::MarkerSet;
using jem::gl::PointSampler;
using jive::util::PointSet;


//=======================================================================
//   class PointView::RunData_
//=======================================================================


class PointView::RunData_ : public jem::Collectable
{
 public:

  typedef RunData_          Self;
  typedef
    PointSampler::Sample    Sample;

  static const int          NEW_COORDS = 1 << 0;
  static const int          NEW_POINTS = 1 << 1;
  static const int          NEW_GROUPS = 1 << 2;


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
      const Properties&       viewdat,
      const AppContext&       apx );

  void                      getConfig

    ( const Properties&       myProps )        const;


 public:

  const String              myContext;


 private:

  void                      coordsChanged_  ();
  void                      pointsChanged_  ();
  void                      groupsChanged_  ();

  void                      pointSampled_

    ( RenderContext&          rtx,
      const Sample&           sample );

  void                      updatePIDs_

    ( const AppContext&       apx );

  void                      updateCoords_

    ( const Properties&       globdat,
      const AppContext&       apx );

  void                      updateMarkers_

    ( const Properties&       globdat,
      const AppContext&       apx );

  void                      selectGroups_

    ( const StringVector&     specs,
      const AppContext&       apx );

  void                      updateGroup_

    ( idx_t                   i,
      const Ref<ItemGroup>&   group,
      const AppContext&       apx );


 private:

  AppRef<PointSet>          points_;
  IdxVector                 pointIDs_;
  String                    pointName_;

  Ref<GroupNode>            gnode_;
  Ref<VertexSet>            vertices_;
  Ref<MarkerSet>            markers_;
  Ref<MarkerStyleTable>     styles_;
  Ref<VectorField>          dispField_;

  StringVector              groupNames_;
  IdxVector                 groupStyles_;

  Array <
    AppRef<ItemGroup> >     groups_;

  int                       dirtyBits_;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline PointView::RunData_::RunData_ ( const String& ctx ) :

  myContext ( ctx )

{
  dirtyBits_ = 0;
}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


void PointView::RunData_::init

  ( const String&      myName,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  viewdat,
    const AppContext&  apx )

{
  using jem::dynamicCast;
  using jem::gl::addChild;
  using jem::gl::GfxNode;
  using jem::gl::VertexSet;
  using jem::util::connect;

  Properties  myConf  = conf   .makeProps ( myName );
  Properties  myProps = props  .findProps ( myName );

  Properties  globdat = viewdat.getProps  ( Viewdat::GLOBDAT );
  Properties  config0 = viewdat.makeProps ( Viewdat::CONFIG0 );

  Ref<GfxNode>       world;
  Ref<GroupNode>     group;
  Ref<PointSampler>  psamp;
  String             name;


  viewdat.get ( world, Viewdat::WORLD_NODE );

  myProps.get ( name, PropNames::POINTS );
  myConf .set ( PropNames::POINTS, name );

  points_[apx] =

    dynamicCast<PointSet> ( ItemSet::find( name, globdat ) );

  if ( ! points_[apx] )
  {
    throw IllegalInputException (
      myContext,
      "undefined point set: " + name
    );
  }

  dirtyBits_ = 0;
  pointName_ = points_[apx]->getItemName   ();

  gnode_     = newInstance<GroupNode> ( myName );
  vertices_  = newInstance<VertexSet> ();
  markers_   = newInstance<MarkerSet> ( myName, vertices_ );
  styles_    = MarkerStyleTable::get  ( viewdat );

  updatePIDs_   ( apx );
  updateCoords_ ( globdat, apx );

  markers_->configure ( props   );
  markers_->getConfig ( conf    );
  markers_->getConfig ( config0 );

  if ( ! myProps.contains( PropNames::MARKERS ) )
  {
    myProps.set ( PropNames::MARKERS, "all" );
  }

  configure ( myProps, viewdat, apx );
  getConfig ( myConf );

  if ( dirtyBits_ & NEW_GROUPS )
  {
    updateMarkers_ ( globdat, apx );

    dirtyBits_ &= ~NEW_GROUPS;
  }

  psamp = newInstance<PointSampler> ( myName + ".pointSampler",
                                      markers_ );

  gnode_->addChild ( markers_ );
  gnode_->addChild ( psamp    );

  addChild ( *world, gnode_ );

  points_[apx]->resetEvents ();

  connect ( points_[apx]->newGeomEvent,
            this, & Self::coordsChanged_ );

  connect ( points_[apx]->newSizeEvent,
            this, & Self::pointsChanged_ );

  connect ( points_[apx]->newOrderEvent,
            this, & Self::pointsChanged_ );

  connect ( psamp->sampleEvent, this, & Self::pointSampled_ );
}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void PointView::RunData_::update

  ( const Properties&  viewdat,
    const AppContext&  apx )

{
  Properties  globdat = viewdat.getProps ( Viewdat::GLOBDAT );


  if ( dirtyBits_ & (NEW_POINTS | NEW_GROUPS) )
  {
    markers_->unmarkAll ();
  }

  if ( dirtyBits_ & NEW_POINTS )
  {
    updatePIDs_ ( apx );
  }

  if ( (dirtyBits_ & NEW_COORDS) || dispField_ )
  {
    updateCoords_ ( globdat, apx );
  }

  if ( dirtyBits_ & (NEW_POINTS | NEW_GROUPS) )
  {
    updateMarkers_ ( globdat, apx );
  }

  points_[apx]->resetEvents ();

  dirtyBits_ = 0;
}


//-----------------------------------------------------------------------
//   shutdown
//-----------------------------------------------------------------------


inline void PointView::RunData_::shutdown ()
{
  jem::gl::eraseFromParent ( gnode_ );
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void PointView::RunData_::configure

  ( const Properties&  myProps,
    const Properties&  viewdat,
    const AppContext&  apx )

{
  using jem::util::isNone;

  StringVector  specs;
  Ref<Object>   obj;


  if ( myProps.find( specs, PropNames::MARKERS ) )
  {
    try
    {
      selectGroups_ ( specs, apx );
    }
    catch ( IllegalInputException& ex )
    {
      ex.setContext ( myProps.getContext( PropNames::MARKERS ) );
      throw;
    }
  }

  if ( ! myProps.find( obj, PropNames::DISPLACEMENTS ) )
  {
    return;
  }

  if ( isNone( obj ) )
  {
    dispField_ = nullptr;
    return;
  }

  if ( ! dispField_ )
  {
    String   compNames[3] = { "dx", "dy", "dz" };
    idx_t    rank         = jem::min ( 3_idx, points_[apx]->rank() );

    dispField_ =

      newInstance<VectorField> ( rank, points_[apx], compNames );
  }

  dispField_->configure ( PropNames::DISPLACEMENTS, myProps,
                          viewdat.findProps( Viewdat::DATA_SETS ) );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void PointView::RunData_::getConfig

  ( const Properties&  myProps ) const

{
  using jem::util::None;

  const idx_t   n = groupNames_.size ();

  StringVector  specs ( n );


  for ( idx_t i = 0; i < n; i++ )
  {
    specs[i] = groupNames_[i] + " : " +
               markers_->getStyleName ( groupStyles_[i] );
  }

  myProps.set ( PropNames::MARKERS, specs );

  if ( dispField_ )
  {
    dispField_->getConfig

      ( myProps.makeProps( PropNames::DISPLACEMENTS ) );
  }
  else
  {
    myProps.set ( PropNames::DISPLACEMENTS,
                  None::getInstance() );
  }
}


//-----------------------------------------------------------------------
//   coordsChanged_
//-----------------------------------------------------------------------


void PointView::RunData_::coordsChanged_ ()
{
  dirtyBits_ |= NEW_COORDS;
}


//-----------------------------------------------------------------------
//   pointsChanged_
//-----------------------------------------------------------------------


void PointView::RunData_::pointsChanged_ ()
{
  dirtyBits_ |= (NEW_COORDS | NEW_POINTS);
}


//-----------------------------------------------------------------------
//   groupsChanged_
//-----------------------------------------------------------------------


void PointView::RunData_::groupsChanged_ ()
{
  dirtyBits_ |= NEW_GROUPS;
}


//-----------------------------------------------------------------------
//   pointSampled_
//-----------------------------------------------------------------------


void PointView::RunData_::pointSampled_

  ( RenderContext&  rtx,
    const Sample&   sample )

{
  using jem::Time;
  using jem::StringBuffer;
  using jem::gl::LEFT_ALIGNED;
  using jem::gl::BOT_ALIGNED;
  using jem::gl::showMessageUntil;

  GLfloat  dist;
  idx_t    ipnt;


  vertices_->findVertex ( ipnt,  dist,
                          (GLfloat) sample.point[0],
                          (GLfloat) sample.point[1],
                          (GLfloat) sample.point[2] );

  if ( ipnt < 0 )
  {
    return;
  }

  const GLfloat*  v   = vertices_->getData () + (3 * ipnt);
  const idx_t     pid = pointIDs_[ipnt];

  StringBuffer    text;
  idx_t           width;

  width = pointName_.size ();

  if ( width < 7 )
  {
    width = 7;
  }

  text.reserve ( 64 );

  text += String::format ( "%*S : %9d\n", -width, pointName_, pid );
  text += String::format ( "%*s : %9.2e\n",
                           -width, "x-coord", (double) v[0] );
  text += String::format ( "%*s : %9.2e\n",
                           -width, "y-coord", (double) v[1] );
  text += String::format ( "%*s : %9.2e",
                           -width, "z-coord", (double) v[2] );

  showMessageUntil ( rtx, gnode_,  text.toString(),
                     Time::now() + 8_sec,
                     LEFT_ALIGNED, BOT_ALIGNED );
}


//-----------------------------------------------------------------------
//   updatePIDs_
//-----------------------------------------------------------------------


void PointView::RunData_::updatePIDs_ ( const AppContext& apx )
{
  const idx_t  pcount = points_[apx]->size ();

  IdxVector    ipoints ( jem::iarray( pcount) );


  pointIDs_.resize ( pcount );

  points_[apx]->getItemMap()->getItemIDs ( pointIDs_, ipoints );
}


//-----------------------------------------------------------------------
//   updateCoords_
//-----------------------------------------------------------------------


void PointView::RunData_::updateCoords_

  ( const Properties&  globdat,
    const AppContext&  apx )

{
  using jem::System;
  using jive::geom::sizeofBox;
  using jive::geom::getBoundingBox;

  GLfloatMatrix  vcoords = vertices_   ->getCoords ();
  Matrix         pcoords = points_[apx]->toMatrix  ();

  const idx_t    pcount  = pcoords.size (1);

  idx_t          i, j, k;


  if ( vcoords.size(0) != 3 || vcoords.size(1) != pcount )
  {
    vcoords.resize ( 3, pcount );
  }

  k = pcoords.size (0);

  if ( k > 3 )
  {
    pcoords.ref ( pcoords(slice(BEGIN,3),ALL) );

    k = 3;
  }

  vcoords = (GLfloat) 0.0;

  for ( j = 0; j < pcount; j++ )
  {
    for ( i = 0; i < k; i++ )
    {
      vcoords(i,j) = (GLfloat) pcoords(i,j);
    }
  }

  if ( dispField_ )
  {
    Matrix  disp ( 3, pcount );

    double  scale;


    if ( dispField_->autoScale )
    {
      Vector  bbox ( 2 * k );

      getBoundingBox ( bbox, pcoords );

      dispField_->setPostScale ( sizeofBox( bbox ) );
    }

    try
    {
      dispField_->getData ( scale, disp, globdat, myContext, apx );

      for ( j = 0; j < pcount; j++ )
      {
        for ( i = 0; i < 3; i++ )
        {
          vcoords(i,j) += (GLfloat) disp(i,j);
        }
      }
    }
    catch ( const jem::IllegalInputException& ex )
    {
      print ( System::warn(), ex, '\n' );
    }
  }

  vertices_->setCoords ( vcoords );
}


//-----------------------------------------------------------------------
//   updateMarkers_
//-----------------------------------------------------------------------


void PointView::RunData_::updateMarkers_

  ( const Properties&  globdat,
    const AppContext&  apx )

{
  using jem::System;

  Ref<ItemGroup>  group;

  IdxVector       pids;


  markers_->unmarkAll ();

  for ( idx_t i = 0; i < groups_.size(); i++ )
  {
    group = ItemGroup::find ( groupNames_[i],
                              points_[apx], globdat );

    if ( groups_[i][apx] != group )
    {
      updateGroup_ ( i, group, apx );
    }

    if ( ! group )
    {
      print ( System::warn(), myContext,
              " : undefined ", pointName_,
              " group: ", groupNames_[i], " (skipping)\n" );

      continue;
    }

    IdxVector  ipoints = group->getIndices ();

    pids.resize ( ipoints.size() );

    pids = pointIDs_[ipoints];

    markers_->markVertices ( ipoints, groupStyles_[i], pids );
    group   ->resetEvents  ();
  }
}


//-----------------------------------------------------------------------
//   selectGroups_
//-----------------------------------------------------------------------


void PointView::RunData_::selectGroups_

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
      styleName   = "orbs";
    }
    else
    {
      newNames[i] = s[slice(BEGIN,j)]  .stripWhite ();
      styleName   = s[slice(j + 1,END)].stripWhite ();
    }

    if ( ! newNames[i].size() )
    {
      throw IllegalInputException (
        myContext,
        String::format ( "empty %s group name", pointName_ )
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

  for ( i = 0; i < groups_.size(); i++ )
  {
    if ( groups_[i][apx] )
    {
      disconnect ( groups_[i][apx]->newSizeEvent,
                   this, & Self::groupsChanged_ );

      groups_[i][apx] = nullptr;
    }
  }

  groupNames_ .swap ( newNames  );
  groupStyles_.swap ( newStyles );

  groups_.reshape   ( n );

  dirtyBits_ |= NEW_GROUPS;
}


//-----------------------------------------------------------------------
//   updateGroup_
//-----------------------------------------------------------------------


void PointView::RunData_::updateGroup_

  ( idx_t                  i,
    const Ref<ItemGroup>&  group,
    const AppContext&      apx )

{
  using jem::util::connect;
  using jem::util::disconnect;

  if ( groups_[i][apx] )
  {
    disconnect ( groups_[i][apx]->newSizeEvent,
                 this, & Self::groupsChanged_ );
  }

  groups_[i][apx] = group;

  if ( group )
  {
    connect    ( group->newSizeEvent,
                 this, & Self::groupsChanged_ );
  }
}


//=======================================================================
//   class PointView
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  PointView::TYPE_NAME = "PointView";


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


PointView::PointView ( const String& name ) :

  Super ( name )

{}


PointView::~PointView ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


void PointView::init

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


void PointView::update

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


void PointView::shutdown

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


void PointView::configure

  ( const Properties&  props,
    const Properties&  viewdat,
    const AppContext&  apx )

{
  if ( rundat_ && props.contains( myName_ ) )
  {
    rundat_->configure ( props.findProps( myName_ ),
                         viewdat, apx );
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void PointView::getConfig ( const Properties& props ) const
{
  if ( rundat_ )
  {
    rundat_->getConfig ( props.makeProps( myName_ ) );
  }
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void PointView::declare ()
{
  PluginFactory::declare ( TYPE_NAME,  & makeNew );
  PluginFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Plugin> PointView::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props )

{
  return newInstance<Self> ( name );
}


JIVE_END_PACKAGE( gl )
