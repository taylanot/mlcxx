
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jem, a general purpose programming toolkit.
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
 *
 *  Jem version: 3.0
 *  Date:        Fri 20 Dec 14:27:58 CET 2019
 */


#include <cstdio>
#include <jem/base/assert.h>
#include <jem/base/System.h>
#include <jem/base/ClassTemplate.h>
#include <jem/util/Properties.h>
#include <jem/gl/utilities.h>
#include <jem/gl/Box.h>
#include <jem/gl/Font.h>
#include <jem/gl/Menu.h>
#include <jem/gl/Actions.h>
#include <jem/gl/Commands.h>
#include <jem/gl/Palette.h>
#include <jem/gl/GLContext.h>
#include <jem/gl/RedrawRoot.h>
#include <jem/gl/RenderContext.h>
#include <jem/gl/Transformable.h>
#include <jem/gl/PalettePainter.h>


JEM_DEFINE_CLASS( jem::gl::PalettePainter );


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class PalettePainter
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*    PalettePainter::PCOUNT_     = "pcount";
const char*    PalettePainter::GET_PCOUNT_ = "_GET_PALETTE_COUNT";
const char*    PalettePainter::SET_PCOUNT_ = "_SET_PALETTE_COUNT";

const GLsizei  PalettePainter::MARGIN_     = 12;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


PalettePainter::PalettePainter

  ( const String&        name,
    const Ref<Palette>&  pal,
    const Ref<GfxNode>&  buddy ) :

    Super    ( name ),
    palette_ ( pal ),
    buddy_   ( buddy )

{
  JEM_PRECHECK ( pal );

  hidden  =  false;
  scale_  =  1.0;
  bias_   =  0.0;
  hAlign_ =  LEFT_ALIGNED;
  vAlign_ =  TOP_ALIGNED;
  pcount_ = -1;
  hide_   =  true;
}


PalettePainter::~PalettePainter ()
{}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool PalettePainter::takeAction

  ( const String&      action,
    const Properties&  params )

{
  if ( action == GET_PCOUNT_ )
  {
    int  pcount;

    if ( isVisible_() )
    {
      if ( ! myHash_.size() )
      {
        initMyHash_ ();
      }

      if ( params.contains( myHash_ ) )
      {
        hide_ = true;
      }
      else
      {
        hide_ = false;

        params.get ( pcount,  PCOUNT_  );
        pcount++;
        params.set ( PCOUNT_, pcount   );
        params.set ( myHash_, palette_ );
      }
    }

    return true;
  }

  if ( action == SET_PCOUNT_ )
  {
    params.get ( pcount_, PCOUNT_ );

    return true;
  }

  if ( action == Actions::HIDE )
  {
    hidden = true;
    return   true;
  }

  if ( action == Actions::SHOW )
  {
    hidden = false;
    return   true;
  }

  if ( action == Actions::LIST_OPTIONS )
  {
    Ref<Menu>  menu;

    params.get ( menu, ActionParams::OPTIONS );

    menu->addOption ( Commands::HIDE_PALETTE,
                      Commands::SHOW_PALETTE, hidden );

    return true;
  }

  return false;
}


//-----------------------------------------------------------------------
//   takeGLAction
//-----------------------------------------------------------------------


bool PalettePainter::takeGLAction

  ( RenderContext&     rtx,
    int                action,
    const Properties&  params )

{
  bool    result = true;
  double  lod;


  switch ( action )
  {
  case GLActions::REDRAW_FAST:

    lod = 1.0;

    params.find ( lod, GLActionParams::DETAIL_LEVEL );

    if ( lod < 0.5 )
    {
      // Do not draw the palette.

      break;
    }

    // Fall through to the redraw procedure.

  case GLActions::REDRAW:

    if ( hidden )
    {
      return false;
    }

    if ( ! buddy_ )
    {
      drawNoBuddy_ ( rtx );
    }
    else
    {
      if ( pcount_ < 0 )
      {
        findPeers_ ();
      }

      if ( ! hide_ )
      {
        drawWithBuddy_ ( rtx );
      }

      pcount_ = -1;
      hide_   =  true;
    }

    break;

  case GLActions::INIT_VIEW:

    {
      int  x, y, w, h;

      params.get ( x, GLActionParams::VIEW_XPOS   );
      params.get ( y, GLActionParams::VIEW_YPOS   );
      params.get ( w, GLActionParams::VIEW_WIDTH  );
      params.get ( h, GLActionParams::VIEW_HEIGHT );

      vport_.x      = (GLint)   x;
      vport_.y      = (GLint)   y;
      vport_.width  = (GLsizei) w;
      vport_.height = (GLsizei) h;
    }

    break;

  case GLActions::INIT_GL:

    pcount_ = -1;

    break;

  case GLActions::EXE_COMMAND:

    {
      String  cmd;

      params.get ( cmd, GLActionParams::COMMAND );

      if       ( cmd == Commands::HIDE_PALETTE )
      {
        hidden = true;
      }
      else if ( cmd == Commands::SHOW_PALETTE )
      {
        hidden = false;
      }
      else
      {
        result = false;
      }
    }

    break;

  default:

    result = false;
  }

  return result;
}


//-----------------------------------------------------------------------
//   getVisibility
//-----------------------------------------------------------------------


void PalettePainter::getVisibility

  ( bool&             viz,
    Box&              vbox,
    const Transform&  proj ) const

{
  viz = true;
  vbox.makeEmpty ();
}


//-----------------------------------------------------------------------
//   setPalette
//-----------------------------------------------------------------------


void PalettePainter::setPalette ( const Ref<Palette>& pal )
{
  JEM_PRECHECK ( pal );

  palette_ = pal;
  myHash_  = "";
}


//-----------------------------------------------------------------------
//   setScaleAndBias
//-----------------------------------------------------------------------


void PalettePainter::setScaleAndBias

  ( double  scale,
    double  bias )

{
  JEM_PRECHECK ( scale > 0.0 );

  scale_  = scale;
  bias_   = bias;
  myHash_ = "";
}


//-----------------------------------------------------------------------
//   setAlignments
//-----------------------------------------------------------------------


void PalettePainter::setAlignments

  ( Alignment  hAlign,
    Alignment  vAlign )

{
  if ( hAlign == CENTERED )
  {
    hAlign = LEFT_ALIGNED;
  }

  hAlign_ = hAlign;
  vAlign_ = vAlign;
}


//-----------------------------------------------------------------------
//   findPeers_
//-----------------------------------------------------------------------


void PalettePainter::findPeers_ ()
{
  GfxNode*  root = findRedrawRoot ( getParent() );

  if ( root )
  {
    Properties  params ( "params" );

    int         pcount  = 0;

    params.set ( PCOUNT_, pcount );

    root->takeAction ( GET_PCOUNT_, params );
    root->takeAction ( SET_PCOUNT_, params );
  }
  else
  {
    pcount_ = 1;
  }

  if ( pcount_ < 0 )
  {
    print ( System::warn(), myName_,
            " : failed to find peer nodes; "
            "botched up scene graph?\n" );
  }
}


//-----------------------------------------------------------------------
//   initMyHash_
//-----------------------------------------------------------------------


void PalettePainter::initMyHash_ ()
{
  char   buf[64];

  idx_t  k = std::sprintf ( buf, "%p%.2e%.2e",
                            palette_.get(), scale_, bias_ );

  myHash_ = String ( buf, k );
}


//-----------------------------------------------------------------------
//   isVisisble_
//-----------------------------------------------------------------------


bool PalettePainter::isVisible_ () const
{
  if ( ! buddy_ )
  {
    return true;
  }
  else
  {
    Transform  p;
    Vertex3d   a, b;
    Box        box;


    buddy_->getBBox ( box );

    if ( box.isEmpty() )
    {
      return false;
    }

    p = getProjectionOf ( buddy_->getParent() );

    box.project    ( p );
    box.getCorners ( a, b );

    if ( a[0] > 1.0 || b[0] < -1.0 ||
         a[1] > 1.0 || b[1] < -1.0 ||
         a[2] > 1.0 || b[2] < -1.0 )
    {
      return false;
    }
  }

  return true;
}


//-----------------------------------------------------------------------
//   drawNoBuddy_
//-----------------------------------------------------------------------


void PalettePainter::drawNoBuddy_ ( RenderContext& rtx )
{
  Font  f;
  Rect  r;


  r.height = (GLsizei) (20 * f.getHeight());

  if ( ! initDrawArea_( r ) )
  {
    return;
  }

  if ( hAlign_ == LEFT_ALIGNED )
  {
    r.x = (GLint) MARGIN_;
  }
  else
  {
    r.x = (GLint) (vport_.width - r.width - MARGIN_);
  }

  if      ( vAlign_ == BOT_ALIGNED )
  {
    r.y = (GLint) MARGIN_;
  }
  else if ( vAlign_ == CENTERED )
  {
    r.y = vport_.y + (GLint) (vport_.height - r.height) / 2;
  }
  else
  {
    r.y = vport_.y + (GLint) (vport_.height - r.height - MARGIN_);
  }

  palette_->render ( rtx.getGLContext(), r, title,
                     hAlign_, scale_, bias_ );
}


//-----------------------------------------------------------------------
//   drawWithBuddy_
//-----------------------------------------------------------------------


void PalettePainter::drawWithBuddy_ ( RenderContext& rtx )
{
  Transform  p;
  Vertex3d   a, b;
  Box        box;
  Rect       r;
  GLdouble   y;


  buddy_->getBBox ( box );

  if ( box.isEmpty() )
  {
    return;
  }

  p = getProjectionOf ( buddy_->getParent() );

  box.project    ( p );
  box.getCorners ( a, b );

  // Forget about it if the object is not visible

  if ( a[0] > 1.0 || b[0] < -1.0 ||
       a[1] > 1.0 || b[1] < -1.0 ||
       a[2] > 1.0 || b[2] < -1.0 )
  {
    return;
  }

  if ( pcount_ <= 1 )
  {
    drawNoBuddy_ ( rtx );
    return;
  }

  r.height = (GLsizei) (vport_.height / 3.0);

  if ( ! initDrawArea_( r ) )
  {
    return;
  }

  y    = 0.5 * (a[1] + b[1]);
  r.y  = vport_.y + (GLint) (0.5 * (y + 1.0) * vport_.height);
  r.y -= (r.height / 2);

  if ( hAlign_ == LEFT_ALIGNED )
  {
    r.x  = vport_.x + (GLint) (0.5 * (a[0] + 1.0) * vport_.width);
    r.x -= 2 * MARGIN_;

    // Forget about it if the palette is not visible

    if ( r.x < vport_.x )
    {
      return;
    }

    r.x -= r.width;

    palette_->render ( rtx.getGLContext(), r, title,
                       RIGHT_ALIGNED, scale_, bias_ );
  }
  else
  {
    r.x  = vport_.x + (GLint) (0.5 * (b[0] + 1.0) * vport_.width);
    r.x += 2 * MARGIN_;

    // Forget about it if the palette is not visible

    if ( r.x > (GLint) (vport_.x + vport_.width) )
    {
      return;
    }

    palette_->render ( rtx.getGLContext(), r, title,
                       LEFT_ALIGNED, scale_,  bias_ );
  }
}


//-----------------------------------------------------------------------
//   initDrawArea_
//-----------------------------------------------------------------------


bool PalettePainter::initDrawArea_ ( Rect& r )
{
  const GLsizei  MARGIN2 = 2 * MARGIN_;


  clamp ( r.height, (GLsizei) 100, (GLsizei) 500 );

  r.height += MARGIN2;

  if ( r.height > vport_.height )
  {
    r.height = vport_.height;
  }

  if ( r.height < (100 + MARGIN2) )
  {
    return false;
  }

  r.height -= MARGIN2;

  r.width   = (GLsizei) (r.height / 4.0);

  clamp     ( r.width, (GLsizei) 50, (GLsizei) 200 );

  r.width  += MARGIN2;

  if ( r.width > vport_.width / 4 )
  {
    r.width = vport_.width / 4;
  }

  if ( r.width < (50 + MARGIN2) )
  {
    return false;
  }

  r.width -= MARGIN2;

  return true;
}


JEM_END_PACKAGE( gl )
