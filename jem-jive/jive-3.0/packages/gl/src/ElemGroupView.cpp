
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
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalInputException.h>
#include <jem/base/array/select.h>
#include <jem/gl/byte.h>
#include <jem/gl/Compound.h>
#include <jem/gl/MessageBox.h>
#include <jem/gl/PointSampler.h>
#include <jem/util/Event.h>
#include <jem/util/Properties.h>
#include <jive/util/utilities.h>
#include <jive/util/ItemGroup.h>
#include <jive/fem/ElementGroup.h>
#include <jive/gl/Names.h>
#include <jive/gl/AppRef.h>
#include <jive/gl/GLArray.h>
#include <jive/gl/Viewdat.h>
#include <jive/gl/MeshRenderer.h>
#include <jive/gl/PluginFactory.h>
#include <jive/gl/ElemGroupView.h>


JEM_DEFINE_CLASS( jive::gl::ElemGroupView );


JIVE_BEGIN_PACKAGE( gl )


using jem::newInstance;
using jem::Array;
using jem::util::Event;
using jem::gl::PointSampler;


//=======================================================================
//   class ElemGroupView::RunData_
//=======================================================================


class ElemGroupView::RunData_ : public jem::Collectable
{
 public:

  typedef RunData_          Self;
  typedef
    PointSampler::Sample    Sample;


  explicit inline           RunData_

    ( const String&           ctx );

  void                      init

    ( const String&           myName,
      const Properties&       myProps,
      const Properties&       myConf,
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

  void                      pointSampled_

    ( RenderContext&          rtx,
      const Sample&           sample );

  void                      selectElems_

    ( const StringVector&     specs,
      const AppContext&       apx );

  void                      updateGroup_

    ( idx_t                   i,
      const Ref<ItemGroup>&   group,
      const AppContext&       apx );


 private:

  Event < RenderContext&,
          const Sample& >   sampleEvent_;

  Ref<MeshRenderer>         mrender_;
  Ref<PointSampler>         sampler_;

  StringVector              groupNames_;
  GLubyteMatrix             groupColors_;

  Array <
    AppRef<ItemGroup> >     groups_;
  IdxVector                 elemGroupMap_;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline ElemGroupView::RunData_::RunData_ ( const String& ctx ) :

  myContext ( ctx )

{
  updated = false;
}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


void ElemGroupView::RunData_::init

  ( const String&      myName,
    const Properties&  myConf,
    const Properties&  myProps,
    const Properties&  viewdat,
    const AppContext&  apx )

{
  using jem::gl::addChild;
  using jem::gl::GfxNode;
  using jem::util::connect;

  Ref<GfxNode>    world;
  Ref<VertexSet>  vertices;


  viewdat.get ( world, Viewdat::WORLD_NODE );

  mrender_ = MeshRenderer::get ( viewdat, myContext );

  configure ( myProps, apx );
  getConfig ( myConf );

  if ( ! updated )
  {
    update ( viewdat, apx );
  }

  sampler_ = PointSampler::find ( world, mrender_ );

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

  connect ( sampler_->sampleEvent, this, & Self::pointSampled_ );
}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void ElemGroupView::RunData_::update

  ( const Properties&  viewdat,
    const AppContext&  apx )

{
  using jem::System;
  using jem::gl::toUbyte;
  using jem::gl::Color;

  Properties      globdat    = viewdat.getProps ( Viewdat::GLOBDAT );

  ElementSet      elems      = mrender_->getElements    ( apx );
  IdxVector       ielems     = mrender_->getElemIndices ( apx );
  Color           meshColor  = mrender_->getColor       ();

  const idx_t     elemCount  = elems .size ();
  const idx_t     ielemCount = ielems.size ();

  Ref<ItemGroup>  group;

  GLubyteMatrix   elemColors ( 4, ielemCount );
  IdxVector       elemMap    (     elemCount );


  elemGroupMap_.resize ( ielemCount );

  elemMap = -1;

  for ( idx_t i = 0; i < groups_.size(); i++ )
  {
    group = ItemGroup::find ( groupNames_[i],
                              elems.getData(), globdat );

    if ( groups_[i][apx] != group )
    {
      updateGroup_ ( i, group, apx );
    }

    if ( ! group )
    {
      print ( System::warn(), myContext,
              " : undefined element group: ",
              groupNames_[i], " (skipping)\n" );

      continue;
    }

    group->resetEvents ();

    elemMap[group->getIndices()] = i;
  }

  elemGroupMap_ = elemMap[ielems];

  for ( idx_t i = 0; i < ielemCount; i++ )
  {
    idx_t  j = elemGroupMap_[i];

    if ( j < 0 )
    {
      elemColors(0,i) = toUbyte ( meshColor[0] );
      elemColors(1,i) = toUbyte ( meshColor[1] );
      elemColors(2,i) = toUbyte ( meshColor[2] );
      elemColors(3,i) = toUbyte ( meshColor[3] );
    }
    else
    {
      elemColors(0,i) = groupColors_(0,j);
      elemColors(1,i) = groupColors_(1,j);
      elemColors(2,i) = groupColors_(2,j);
      elemColors(3,i) = groupColors_(3,j);
    }
  }

  mrender_->setElemColors ( elemColors, apx );

  updated = true;
}


//-----------------------------------------------------------------------
//   shutdown
//-----------------------------------------------------------------------


inline void ElemGroupView::RunData_::shutdown ()
{
  mrender_->clearColors ();
  sampler_->sampleEvent.swap ( sampleEvent_ );
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void ElemGroupView::RunData_::configure

  ( const Properties&  myProps,
    const AppContext&  apx )

{
  StringVector  specs;

  if ( myProps.find( specs, PropNames::ELEMENTS ) )
  {
    try
    {
      selectElems_ ( specs, apx );
    }
    catch ( jem::IllegalInputException& ex )
    {
      ex.setContext ( myProps.getContext( PropNames::ELEMENTS ) );
      throw;
    }
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void ElemGroupView::RunData_::getConfig

  ( const Properties&  myProps ) const

{
  using jem::gl::toFloat;
  using jem::gl::Color;

  const idx_t   n = groupNames_.size ();

  StringVector  specs ( n );

  Color         c;


  for ( idx_t i = 0; i < n; i++ )
  {
    c[0]     = toFloat ( groupColors_(0,i) );
    c[1]     = toFloat ( groupColors_(1,i) );
    c[2]     = toFloat ( groupColors_(2,i) );
    c[3]     = toFloat ( groupColors_(3,i) );

    specs[i] = groupNames_[i] + String ( " : " ) + c.toString ();
  }

  myProps.set ( PropNames::ELEMENTS, specs );
}


//-----------------------------------------------------------------------
//   markOutdated_
//-----------------------------------------------------------------------


void ElemGroupView::RunData_::markOutdated_ ()
{
  updated = false;
}


//-----------------------------------------------------------------------
//   pointSampled_
//-----------------------------------------------------------------------


void ElemGroupView::RunData_::pointSampled_

  ( RenderContext&  rtx,
    const Sample&   sample )

{
  using jem::Time;
  using jem::gl::LEFT_ALIGNED;
  using jem::gl::BOT_ALIGNED;
  using jem::gl::showMessageUntil;

  if ( ! updated )
  {
    sampleEvent_.emit ( rtx, sample );
  }
  else
  {
    String  groupName;
    String  text;

    idx_t   igroup;
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

    igroup = elemGroupMap_[ielem];

    if ( igroup < 0 )
    {
      groupName = "none";
    }
    else
    {
      groupName = groupNames_[igroup];
    }

    text = String::format ( "element    : %10d\n"
                            "x-coord    : %10.2e\n"
                            "y-coord    : %10.2e\n"
                            "z-coord    : %10.2e\n"
                            "group name : %10s",

                            elemID,
                            sample.point[0],
                            sample.point[1],
                            sample.point[2],
                            groupName );

    showMessageUntil ( rtx, mrender_, text,
                       Time::now() +  8_sec,
                       LEFT_ALIGNED,  BOT_ALIGNED );
  }
}


//-----------------------------------------------------------------------
//   selectElems_
//-----------------------------------------------------------------------


void ElemGroupView::RunData_::selectElems_

  ( const StringVector&  specs,
    const AppContext&    apx )

{
  using jem::gl::toUbyte;
  using jem::gl::Color;
  using jem::util::disconnect;

  const idx_t    n = specs.size ();

  StringVector   newNames  (    n );
  GLubyteMatrix  newColors ( 4, n );

  Color          c;


  for ( idx_t i = 0; i < n; i++ )
  {
    String  s = specs[i];
    idx_t   j = s.find ( ':' );

    if ( j < 0 )
    {
      throw jem::IllegalInputException (
        CLASS_NAME,
        "no color specified (missing `:\')"
      );
    }

    newNames[i] = s[slice(BEGIN,j)].stripWhite ();

    if ( ! newNames[i].size() )
    {
      throw jem::IllegalInputException (
        CLASS_NAME,
        "empty element group name"
      );
    }

    s = s[slice(j + 1,END)].stripWhite ();

    if ( ! s.size() )
    {
      throw jem::IllegalInputException (
        CLASS_NAME,
        "no color specified after `:\'"
      );
    }

    c = Color::parse ( s );

    newColors(0,i) = toUbyte ( c[0] );
    newColors(1,i) = toUbyte ( c[1] );
    newColors(2,i) = toUbyte ( c[2] );
    newColors(3,i) = toUbyte ( c[3] );
  }

  for ( idx_t i = 0; i < groups_.size(); i++ )
  {
    if ( groups_[i][apx] )
    {
      disconnect ( groups_[i][apx]->newSizeEvent,
                   this, & Self::markOutdated_ );

      groups_[i] = nullptr;
    }
  }

  groupNames_ .swap ( newNames  );
  groupColors_.swap ( newColors );

  groups_.reshape   ( n );

  updated = false;
}


//-----------------------------------------------------------------------
//   updateGroup_
//-----------------------------------------------------------------------


void ElemGroupView::RunData_::updateGroup_

  ( idx_t                  i,
    const Ref<ItemGroup>&  group,
    const AppContext&      apx  )

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
    connect ( group->newSizeEvent,
              this, & Self::markOutdated_ );
  }
}


//=======================================================================
//   class ElemGroupView
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  ElemGroupView::TYPE_NAME  = "ElemGroupView";


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


ElemGroupView::ElemGroupView ( const String& name ) :

  Super ( name )

{}


ElemGroupView::~ElemGroupView ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


void ElemGroupView::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  viewdat,
    const AppContext&  apx )

{
  rundat_ = nullptr;

  Properties     myConf  = conf .makeProps ( myName_ );
  Properties     myProps = props.findProps ( myName_ );

  Ref<RunData_>  newdat  = newInstance<RunData_> ( getContext() );

  newdat->init ( myName_, myConf, myProps, viewdat, apx );
  rundat_.swap ( newdat );
}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void ElemGroupView::update

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


void ElemGroupView::shutdown

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


void ElemGroupView::configure

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


void ElemGroupView::getConfig ( const Properties& props ) const
{
  if ( rundat_ )
  {
    rundat_->getConfig ( props.makeProps( myName_ ) );
  }
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void ElemGroupView::declare ()
{
  PluginFactory::declare ( TYPE_NAME,  & makeNew );
  PluginFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Plugin> ElemGroupView::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props )

{
  return newInstance<Self> ( name );
}


JIVE_END_PACKAGE( gl )
