
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
#include <jem/base/Time.h>
#include <jem/base/System.h>
#include <jem/base/StringBuffer.h>
#include <jem/base/ClassTemplate.h>
#include <jem/gl/utilities.h>
#include <jem/gl/Compound.h>
#include <jem/gl/ColorMap.h>
#include <jem/gl/PointSampler.h>
#include <jem/gl/MessageBox.h>
#include <jem/gl/PalettePainter.h>
#include <jem/util/Event.h>
#include <jem/util/Properties.h>
#include <jem/util/StringUtils.h>
#include <jive/util/utilities.h>
#include <jive/fem/ElementGroup.h>
#include <jive/gl/Names.h>
#include <jive/gl/Viewdat.h>
#include <jive/gl/MeshRenderer.h>
#include <jive/gl/ColorField.h>
#include <jive/gl/PaletteTable.h>
#include <jive/gl/PluginFactory.h>
#include <jive/gl/MeshColorView.h>


JEM_DEFINE_CLASS( jive::gl::MeshColorView );


JIVE_BEGIN_PACKAGE( gl )


using jem::newInstance;
using jem::util::Event;
using jem::gl::PointSampler;
using jem::gl::PalettePainter;


//=======================================================================
//   class MeshColorView::RunData_
//=======================================================================


class MeshColorView::RunData_ : public jem::Collectable
{
 public:

  typedef RunData_          Self;
  typedef
    PointSampler::Sample    Sample;


  explicit inline           RunData_

    ( const String&           ctx );

  void                      init

    ( const String&           myName,
      const Properties&       myConf,
      const Properties&       myProps,
      const Properties&       viewdat );

  void                      update

    ( const Properties&       globdat,
      const AppContext&       apx );

  inline void               shutdown  ();

  void                      configure

    ( const Properties&       myProps,
      const Properties&       viewdat );

  inline void               getConfig

    ( const Properties&       myConf )   const;


 public:

  const String              myContext;


 private:

  void                      pointSampled_

    ( RenderContext&          rtx,
      const Sample&           sample );


 private:

  Event < RenderContext&,
          const Sample& >   sampleEvent_;

  Ref<MeshRenderer>         mrender_;
  Ref<PointSampler>         sampler_;
  Ref<PaletteTable>         palettes_;
  Ref<ColorField>           colors_;
  Ref<PalettePainter>       palPainter_;

  String                    palName_;
  String                    title_;
  double                    scale_;
  double                    bias_;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline MeshColorView::RunData_::RunData_ ( const String& ctx ) :

  myContext ( ctx )

{
  scale_ = 1.0;
  bias_  = 0.0;
}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


void MeshColorView::RunData_::init

  ( const String&      myName,
    const Properties&  myConf,
    const Properties&  myProps,
    const Properties&  viewdat )

{
  using jem::gl::addChild;
  using jem::gl::GfxNode;
  using jem::gl::Palette;
  using jem::util::connect;
  using jive::util::joinNames;

  Properties    dataSets = viewdat.findProps ( Viewdat::DATA_SETS );

  String        palName  = "rainbow";

  Ref<GfxNode>  world;
  Ref<Palette>  palette;
  String        title;


  viewdat.get ( world, Viewdat::WORLD_NODE );

  mrender_  = MeshRenderer::get ( viewdat, myContext );
  palettes_ = PaletteTable::get ( viewdat );

  myProps.find ( palName, PropNames::PALETTE );
  myConf .set  ( PropNames::PALETTE, palName );

  palette     = palettes_->getPalette   ( palName, myContext );
  colors_     = newInstance<ColorField> ( palette );
  palName_    = palName;

  palPainter_ = newInstance<PalettePainter> (
    joinNames ( myName, PropNames::PALETTE ),
    palette,
    mrender_
  );

  myProps.find ( title_, PropNames::TITLE );
  myProps.find ( palPainter_->hidden, PropNames::HIDE_PALETTE );

  myConf .set  ( PropNames::TITLE, title_ );

  colors_->configure ( myProps, dataSets );
  colors_->getConfig ( myConf );

  addChild   ( *world, palPainter_ );

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


void MeshColorView::RunData_::update

  ( const Properties&  globdat,
    const AppContext&  apx )

{
  using jem::System;
  using jem::gl::isOpaque;
  using jem::gl::ColorMap;
  using jem::util::StringUtils;
  using jive::util::expandString;

  PalettePainter&  pp = * palPainter_;

  ElementSet  elems = mrender_->getElements         ( apx );
  NodeSet     nodes = mrender_->getNodes            ( apx );
  ItemSet*    items = colors_ ->dataSource.getItems ( apx );


  if ( title_.size() == 0 )
  {
    pp.title = colors_->dataSource.toString ();
  }
  else
  {
    pp.title = title_;
  }

  pp.title = expandString ( pp.title, globdat );

  if ( pp.title.size() > 16 )
  {
    pp.title = StringUtils::shorten ( pp.title, 16 );
  }

  scale_ = 1.0;
  bias_  = 0.0;

  if      ( items == nodes.getData() )
  {
    GLfloatVector  colors ( items->size() );
    Vector         data   ( colors.size() );

    colors_->dataSource.getData ( data, globdat, myContext, apx );

    Ref<ColorMap>  cmap =

      colors_->getMappedColors ( scale_, bias_,
                                 colors, data, myContext );

    mrender_ ->setNodeColors   ( colors, cmap, apx );
  }
  else if ( items == elems.getData() )
  {
    IdxVector      ielems = mrender_->getElemIndices ( apx );

    GLubyteMatrix  colors;

    if ( isOpaque( colors_->getMinAlpha() ) )
    {
      colors.resize ( 3, ielems.size() );
    }
    else
    {
      colors.resize ( 4, ielems.size() );
    }

    Vector  data ( colors.size(1) );

    colors_ ->dataSource.getData ( data,    ielems,
                                   globdat, myContext, apx );

    colors_ ->getColors     ( scale_, bias_,
                              colors, data, myContext );
    mrender_->setElemColors ( colors, apx );
  }
  else
  {
    print ( System::warn(), myContext,
            " : color data set is not associated with "
            "nodes or elements\n" );
    return;
  }

  scale_ =    1.0 / scale_;
  bias_  = -bias_ * scale_;

  pp.setScaleAndBias ( scale_, bias_ );
}


//-----------------------------------------------------------------------
//   shutdown
//-----------------------------------------------------------------------


inline void MeshColorView::RunData_::shutdown ()
{
  using jem::gl::eraseFromParent;

  eraseFromParent ( palPainter_ );

  mrender_->clearColors ();
  sampler_->sampleEvent.swap ( sampleEvent_ );
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void MeshColorView::RunData_::configure

  ( const Properties&  myProps,
    const Properties&  viewdat )

{
  using jem::gl::Palette;

  Properties  dataSets = viewdat.findProps ( Viewdat::DATA_SETS );

  String      name;


  if ( myProps.find( name, PropNames::PALETTE ) )
  {
    Ref<Palette>  pal = palettes_->getPalette ( name, myContext );

    colors_    ->setPalette ( pal );
    palPainter_->setPalette ( pal );

    palName_ = name;
  }

  myProps.find ( title_,              PropNames::TITLE );
  myProps.find ( palPainter_->hidden, PropNames::HIDE_PALETTE );

  colors_->configure ( myProps, dataSets );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


inline void MeshColorView::RunData_::getConfig

  ( const Properties&  myConf ) const

{
  myConf.set ( PropNames::PALETTE,      palName_ );
  myConf.set ( PropNames::HIDE_PALETTE, palPainter_->hidden );
  myConf.set ( PropNames::TITLE,        title_ );

  colors_->getConfig ( myConf );
}


//-----------------------------------------------------------------------
//   pointSampled_
//-----------------------------------------------------------------------


void MeshColorView::RunData_::pointSampled_

  ( RenderContext&  rtx,
    const Sample&   sample )

{
  using jem::Time;
  using jem::StringBuffer;
  using jem::gl::LEFT_ALIGNED;
  using jem::gl::BOT_ALIGNED;
  using jem::gl::clamped;
  using jem::gl::showMessageUntil;
  using jem::util::StringUtils;

  idx_t  elemID;
  idx_t  ielem;

  ielem = mrender_->findElement ( elemID,
                                  sample.point[0],
                                  sample.point[1],
                                  sample.point[2] );

  if ( ielem < 0 )
  {
    return;
  }

  String        data = palPainter_->title;

  StringBuffer  text;
  String        value;

  double        xval;
  double        eps;

  int           prec[4];
  idx_t         width;


  text.reserve ( 128 );

  // Determine the number of digits to show for the coordinates.

  for ( int i = 0; i < 3; i++ )
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

  // Determine the data value in the sample point.

  xval = 0.0;
  eps  = 0.01;

  eps  = colors_->getPalette()->findValue ( xval, eps,
                                            sample.color );

  xval = bias_ + scale_ * xval;

  if ( eps > 0.25 )
  {
    value = "unknown";
  }
  else
  {
    if ( eps > 0.0 )
    {
      prec[3] = (int) std::ceil ( -std::log10 ( eps ) );
      prec[3] = clamped ( prec[3], 1, 3 );
    }
    else
    {
      prec[3] = 3;
    }

    value = String::format ( "% .*g", prec[3], xval );
  }

  if ( data.size() == 0 )
  {
    data = colors_->dataSource.toString ();
  }

  if ( data.size() > 16 )
  {
    data = StringUtils::shorten ( data, 16 );
  }

  width = -jem::max ( 8_idx, data.size() );

  text += String::format ( "%*s : % d\n",
                           width, "element", elemID );
  text += String::format ( "%*s : % .*g\n",
                           width, "x-coord", prec[0],
                           sample.point[0] );
  text += String::format ( "%*s : % .*g\n",
                           width, "y-coord", prec[1],
                           sample.point[1] );
  text += String::format ( "%*s : % .*g\n",
                           width, "z-coord", prec[2],
                           sample.point[2] );
  text += String::format ( "%*s : %s",
                           width, data, value );

  showMessageUntil ( rtx, mrender_, text.toString(),
                     Time::now() +  10_sec,
                     LEFT_ALIGNED,  BOT_ALIGNED );
}


//=======================================================================
//   class MeshColorView
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  MeshColorView::TYPE_NAME = "MeshColorView";


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


MeshColorView::MeshColorView ( const String& name ) :

  Super ( name )

{}


MeshColorView::~MeshColorView ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


void MeshColorView::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  viewdat,
    const AppContext&  apx )

{
  rundat_ = nullptr;

  Properties     myConf  = conf .makeProps ( myName_ );
  Properties     myProps = props.getProps  ( myName_ );

  Ref<RunData_>  newdat  = newInstance<RunData_> ( getContext() );

  newdat->init ( myName_, myConf, myProps, viewdat );
  rundat_.swap ( newdat );
}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void MeshColorView::update

  ( const Properties&  viewdat,
    const AppContext&  apx )

{
  if ( rundat_ )
  {
    rundat_->update ( viewdat.getProps( Viewdat::GLOBDAT ), apx );
  }
}


//-----------------------------------------------------------------------
//   shutdown
//-----------------------------------------------------------------------


void MeshColorView::shutdown

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


void MeshColorView::configure

  ( const Properties&  props,
    const Properties&  viewdat,
    const AppContext&  apx )

{
  if ( rundat_ && props.contains( myName_ ) )
  {
    rundat_->configure ( props.findProps( myName_ ), viewdat );
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void MeshColorView::getConfig ( const Properties& props ) const
{
  if ( rundat_ )
  {
    rundat_->getConfig ( props.makeProps( myName_ ) );
  }
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void MeshColorView::declare ()
{
  PluginFactory::declare ( TYPE_NAME,  & makeNew );
  PluginFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Plugin> MeshColorView::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props )

{
  return newInstance<Self> ( name );
}


JIVE_END_PACKAGE( gl )
