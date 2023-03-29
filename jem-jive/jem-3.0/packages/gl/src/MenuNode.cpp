
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


#include <jem/base/ClassTemplate.h>
#include <jem/util/Properties.h>
#include <jem/gl/input.h>
#include <jem/gl/render.h>
#include <jem/gl/Menu.h>
#include <jem/gl/Actions.h>
#include <jem/gl/GLContext.h>
#include <jem/gl/GLFunctions1.h>
#include <jem/gl/RedrawRoot.h>
#include <jem/gl/RenderContext.h>
#include <jem/gl/ShaderDraw.h>
#include <jem/gl/TextPainter.h>
#include <jem/gl/MenuNode.h>


JEM_DEFINE_CLASS( jem::gl::MenuNode );


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class MenuNode
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  MenuNode::HIDE_       = "_HIDE_MENU";
const idx_t  MenuNode::FOCUS_NONE_ = -1;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


MenuNode::MenuNode

  ( const String&  name,
    const Menu&    menu ) :

  Super ( name )

{
  using jem::util::List;

  List<String>::ConstIterator  first = menu.entries_.begin ();
  List<String>::ConstIterator  last  = menu.entries_.end   ();

  items_.resize ( menu.entries_.size() );

  for ( idx_t i = 0; first != last; ++first, i++ )
  {
    items_[i].label = *first;
  }

  space_ = 0;
  focus_ = FOCUS_NONE_;
}


MenuNode::MenuNode

  ( const String&         name,
    const Array<String>&  list ) :

  Super ( name )

{
  const idx_t  n = list.size ();

  items_.resize ( n );

  for ( idx_t i = 0; i < n; i++ )
  {
    items_[i].label = list[i];
  }

  space_ = 0;
  focus_ = FOCUS_NONE_;
}


MenuNode::~MenuNode ()
{}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool MenuNode::takeAction

  ( const String&      action,
    const Properties&  params )

{
  if ( isActive() )
  {
    if ( action == HIDE_ || action == Actions::GIVE_FOCUS )
    {
      hide ();

      return true;
    }
    else
    {
      passAction ( child_, action, params );
    }
  }
  else
  {
    bool  retval = passAction ( child_, action, params );

    if ( action == Actions::SET_FONT )
    {
      font_  = Font::getDefault ();
      retval = true;
    }

    return retval;
  }

  return false;
}


//-----------------------------------------------------------------------
//   takeGLAction
//-----------------------------------------------------------------------


bool MenuNode::takeGLAction

  ( RenderContext&     rtx,
    int                action,
    const Properties&  params )

{
  typedef GLActionParams  Params;

  bool  result = true;

  switch ( action )
  {
  case GLActions::KEY_EVENT:

    {
      int  key;

      params.get ( key, Params::KEY_CODE );

      if      ( key == Keys::UP )
      {
        moveFocus_ ( rtx, 1 );
      }
      else if ( key == Keys::DOWN )
      {
        moveFocus_ ( rtx, -1 );
      }
      else if ( key == '\n' )
      {
        selectFocus_ ( rtx );
      }
      else
      {
        hide ( rtx );
      }
    }

    break;

  case GLActions::MOUSE_EVENT:

    {
      bool  down;

      params.get ( down, Params::MOUSE_DOWN );

      if ( down )
      {
        int  x, y;

        params.get ( x, Params::MOUSE_XPOS );
        params.get ( y, Params::MOUSE_YPOS );

        if ( ! rect_.contains( (GLint) x, (GLint) y ) )
        {
          hide ( rtx );
        }
      }
      else if ( focus_ != FOCUS_NONE_ )
      {
        selectFocus_ ( rtx );
      }
    }

    break;

  case GLActions::MOTION_EVENT:

    {
      int  x, y;

      params.get ( x, Params::MOUSE_XPOS );
      params.get ( y, Params::MOUSE_YPOS );

      setFocus_  ( rtx, (GLint) x, (GLint) y );
    }

    break;

  case GLActions::REDRAW:
  case GLActions::REDRAW_FAST:

    passGLAction ( child_, rtx, action, params );
    redraw_      ( rtx );

    break;

  case GLActions::INIT_VIEW:

    passGLAction ( child_, rtx, action, params );
    hide         ();

    break;

  default:

    result = passGLAction ( child_, rtx, action, params );
  }

  return result;
}


//-----------------------------------------------------------------------
//   setChild
//-----------------------------------------------------------------------


void MenuNode::setChild ( const Ref<GfxNode>& child )
{
  Super::setChild_ ( child );
}


//-----------------------------------------------------------------------
//   getChild
//-----------------------------------------------------------------------


Ref<GfxNode> MenuNode::getChild () const
{
  return child_;
}


//-----------------------------------------------------------------------
//   show
//-----------------------------------------------------------------------


void MenuNode::show

  ( RenderContext&       rtx,
    const Ref<GfxNode>&  node )

{
  Vertex2i  pos = rtx.getMousePos ();

  show ( rtx, node, pos[0], pos[1] );
}


void MenuNode::show

  ( RenderContext&       rtx,
    const Ref<GfxNode>&  node,
    int                  xPos,
    int                  yPos )

{
  if ( isActive() )
  {
    return;
  }

  // Go up the scene graph to find the highest pivotable node that is
  // located below or at the same level as the redraw root.

  GfxNode*    current = node.get ();
  Pivotable*  pivot   = 0;

  while ( current )
  {
    Pivotable*  tmpPivot = dynamic_cast<Pivotable*> ( current );

    if ( tmpPivot )
    {
      pivot = tmpPivot;
    }

    if ( dynamic_cast<RedrawRoot*>( current ) )
    {
      break;
    }

    current = current->getParent ();
  }

  if ( pivot )
  {
    show_ ( rtx, *pivot, xPos, yPos );
  }
}


//-----------------------------------------------------------------------
//   hide
//-----------------------------------------------------------------------


void MenuNode::hide ()
{
  Pivotable*  pivot = dynamic_cast<Pivotable*> ( getParent() );

  if ( pivot )
  {
    Ref<GfxNode>  child = child_;

    Super::setChild_ ( nullptr );
    pivot->setChild  ( child );
  }
}


void MenuNode::hide ( RenderContext& rtx )
{
  if ( isActive() )
  {
    rtx.postRedraw ( this );
    hide           ();
  }
}


//-----------------------------------------------------------------------
//   show_
//-----------------------------------------------------------------------


void MenuNode::show_

  ( RenderContext&  rtx,
    Pivotable&      pivot,
    GLint           xPos,
    GLint           yPos )

{
  Ref<GfxNode>  child = pivot.getChild ();

  Properties    params;

  pivot .setChild    ( this );
  Super::setChild_   ( child );
  child->takeAction  ( Actions::GIVE_INPUT, params );
  child->takeAction  ( HIDE_, params );
  this ->init_       ( rtx, (GLint) xPos, (GLint) yPos );
  rtx.postFastRedraw ( this );
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void MenuNode::init_

  ( RenderContext&  rtx,
    GLint           xPos,
    GLint           yPos )

{
  using jem::max;

  Rect     vport = rtx.getGLContext().getViewport ();

  GLsizei  fontHeight;
  GLsizei  w, h;
  GLint    x, y;


  font_  = Font::getDefault ();
  focus_ = FOCUS_NONE_;

  // Set the font size and determine the menu height.

  do
  {
    fontHeight = font_.getHeight ();
    space_     = fontHeight / 4;

    if ( space_ < 1 )
    {
      space_ = 1;
    }

    w = 0;
    h = space_;

    for ( idx_t i = 0; i < items_.size(); i++ )
    {
      const String&  label = items_[i].label;

      h += space_;

      if ( label.size() == 0 )
      {
        h += 2;
      }
      else
      {
        h += fontHeight;
        w  = max ( w, font_.getWidthOf( label ) );
      }

      h += space_;
    }

    h +=     space_;
    w += 4 * space_;

    if ( w <= vport.width && h <= vport.height )
    {
      break;
    }

    if ( font_.getSize() == SMALL_FONT )
    {
      break;
    }

    font_--;
  }
  while ( true );

  rect_.width  = w;
  rect_.height = h;

  // Determine the menu position.

  x = xPos;
  y = yPos;

  if      ( y - h >= vport.y )
  {
    y = y - h;
  }
  else if ( y + h > vport.y + vport.height )
  {
    y = vport.y;
  }

  if ( x + w > vport.x + vport.width )
  {
    if ( x - w >= vport.x )
    {
      x = x - w;
    }
    else
    {
      x = vport.x;
    }
  }

  rect_.x = x;
  rect_.y = y;

  // Initialize the menu item rectangles.

  x = x + space_;
  y = y + h - space_;
  w = w - 2 * space_;

  for ( idx_t i = 0; i < items_.size(); i++ )
  {
    Item_&  item = items_[i];

    h = 2 * space_;

    if ( item.label.size() == 0 )
    {
      h += 2;
    }
    else
    {
      h += fontHeight;
    }

    y = y - h;

    item.rect.x      = x;
    item.rect.y      = y;
    item.rect.width  = w;
    item.rect.height = h;
  }
}


//-----------------------------------------------------------------------
//   redraw_
//-----------------------------------------------------------------------


void MenuNode::redraw_ ( RenderContext& rtx )
{
  GLContext&   gtx     = rtx.getGLContext ();

  AttribScope  attribs ( gtx, GLContext::COLOR |
                              GLContext::LINE_WIDTH );

  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    redrawV3_ ( rtx );
  }
  else
  {
    redrawV1_ ( rtx );
  }
}


//-----------------------------------------------------------------------
//   redrawV1_
//-----------------------------------------------------------------------


void MenuNode::redrawV1_ ( RenderContext& rtx )
{
  const Color    MENU_COLOR  = Color ( 0.800F, 0.800F, 0.800F );
  const Color    FOCUS_COLOR = Color ( 0.438F, 0.500F, 0.563F );
  const Color    LIGHT_COLOR = Color ( 0.900F, 0.900F, 0.900F );
  const Color    DARK_COLOR  = Color ( 0.500F, 0.500F, 0.500F );

  GLContext&     gtx         = rtx.getGLContext ();
  GLFunctions1*  funcs       = gtx.findFunctions<GLFunctions1> ();

  CanvasScope    canvasScope ( gtx );
  TextPainter    txp         ( font_, LEFT_ALIGNED, BOT_ALIGNED );

  GLsizei        w, h;
  GLint          x, y;


 if ( ! funcs )
  {
    return;
  }

  gtx.setLineWidth ( 1.0F );

  // Draw the background.

  gtx.setColor ( MENU_COLOR );
  renderRect   ( gtx, rect_ );

  if ( focus_ != FOCUS_NONE_ )
  {
    gtx.setColor ( FOCUS_COLOR );
    renderRect   ( gtx, items_[focus_].rect );
  }

  // Draw the borders.

  x = rect_.x;
  y = rect_.y;
  w = rect_.width;
  h = rect_.height;

  funcs->glBegin ( GL_LINES );
  {
    gtx.setColor    ( LIGHT_COLOR );
    funcs->glVertex ( x, y + 1 );
    funcs->glVertex ( x, y + h );
    funcs->glVertex ( x, y + h );
    funcs->glVertex ( x + w, y + h );
    gtx.setColor    ( DARK_COLOR );
    funcs->glVertex ( x, y );
    funcs->glVertex ( x + w, y );
    funcs->glVertex ( x + w, y );
    funcs->glVertex ( x + w, y + h - 1 );
  }
  funcs->glEnd ();

  // Draw the menu items.

  for ( idx_t i = 0; i < items_.size(); i++ )
  {
    const Item_&  item = items_[i];

    x = item.rect.x + space_;
    y = item.rect.y + space_;

    if ( item.label.size() == 0 )
    {
      w = item.rect.width - 2 * space_;

      funcs->glBegin ( GL_LINES );
      {
        gtx.setColor    ( LIGHT_COLOR );
        funcs->glVertex ( x,     y );
        funcs->glVertex ( x + w, y );
        gtx.setColor    ( DARK_COLOR );
        funcs->glVertex ( x,     y + 1 );
        funcs->glVertex ( x + w, y + 1 );
      }
      funcs->glEnd ();
    }
    else
    {
      if ( i == focus_ )
      {
        gtx.setColor ( Color::white() );
      }
      else
      {
        gtx.setColor ( Color::black() );
      }

      txp.begin     ( gtx );
      txp.setCursor ( x, y );
      txp.writeLine ( gtx, item.label );
      txp.end       ( gtx );
    }
  }
}


//-----------------------------------------------------------------------
//   redrawV3_
//-----------------------------------------------------------------------


void MenuNode::redrawV3_ ( RenderContext& rtx )
{
  const size_t   COLOR_SIZE  = 4 * sizeof ( GLfloat );
  const Color    MENU_COLOR  = Color ( 0.800F, 0.800F, 0.800F );
  const Color    FOCUS_COLOR = Color ( 0.438F, 0.500F, 0.563F );
  const Color    LIGHT_COLOR = Color ( 0.900F, 0.900F, 0.900F );
  const Color    DARK_COLOR  = Color ( 0.500F, 0.500F, 0.500F );

  GLContext&     gtx         = rtx.getGLContext ();

  GLfloat        vertPos[24] = { 0.0F };
  GLfloat        vertCol[32];

  CanvasScope    canvasScope ( gtx );
  TextPainter    txp         ( font_, LEFT_ALIGNED, BOT_ALIGNED );

  GLfloat        w, h;
  GLfloat        x, y;


  gtx.setLineWidth ( 1.0F );

  // Draw the background.

  gtx.setColor ( MENU_COLOR );
  renderRect   ( gtx, rect_ );

  if ( focus_ != FOCUS_NONE_ )
  {
    gtx.setColor ( FOCUS_COLOR );
    renderRect   ( gtx, items_[focus_].rect );
  }

  // Draw the borders.

  x = (GLfloat) rect_.x;
  y = (GLfloat) rect_.y;
  w = (GLfloat) rect_.width;
  h = (GLfloat) rect_.height;

  vertPos[ 0] = x;     vertPos[ 1] = y + 1;
  vertPos[ 3] = x;     vertPos[ 4] = y + h;
  vertPos[ 6] = x;     vertPos[ 7] = y + h;
  vertPos[ 9] = x + w; vertPos[10] = y + h;

  memcpy ( &vertCol[ 0], LIGHT_COLOR.addr(), COLOR_SIZE );
  memcpy ( &vertCol[ 4], LIGHT_COLOR.addr(), COLOR_SIZE );
  memcpy ( &vertCol[ 8], LIGHT_COLOR.addr(), COLOR_SIZE );
  memcpy ( &vertCol[12], LIGHT_COLOR.addr(), COLOR_SIZE );

  vertPos[12] = x;     vertPos[13] = y;
  vertPos[15] = x + w; vertPos[16] = y;
  vertPos[18] = x + w; vertPos[19] = y;
  vertPos[21] = x + w; vertPos[22] = y + h - 1;

  memcpy ( &vertCol[16], LIGHT_COLOR.addr(), COLOR_SIZE );
  memcpy ( &vertCol[20], LIGHT_COLOR.addr(), COLOR_SIZE );
  memcpy ( &vertCol[24], LIGHT_COLOR.addr(), COLOR_SIZE );
  memcpy ( &vertCol[28], LIGHT_COLOR.addr(), COLOR_SIZE );

  ShaderDraw::draw ( gtx, 24, vertPos, vertCol, ShaderDraw::LINES );

  // Draw the menu items.

  memcpy ( &vertCol[ 0], LIGHT_COLOR.addr(), COLOR_SIZE );
  memcpy ( &vertCol[ 4], LIGHT_COLOR.addr(), COLOR_SIZE );
  memcpy ( &vertCol[ 8], DARK_COLOR.addr(),  COLOR_SIZE );
  memcpy ( &vertCol[12], DARK_COLOR.addr(),  COLOR_SIZE );

  for ( idx_t i = 0; i < items_.size(); i++ )
  {
    const Item_&  item = items_[i];

    if ( item.label.size() == 0 )
    {
      x = (GLfloat)(item.rect.x + space_);
      y = (GLfloat)(item.rect.y + space_);
      w = (GLfloat)(item.rect.width - 2 * space_);

      vertPos[ 0] = x;     vertPos[ 1] = y;
      vertPos[ 3] = x + 2; vertPos[ 4] = y;
      vertPos[ 6] = x;     vertPos[ 7] = y + 1;
      vertPos[ 9] = x + w; vertPos[10] = y + 1;

      ShaderDraw::draw ( gtx, 12, vertPos, vertCol, ShaderDraw::LINES );
    }
    else
    {
      GLint  x = item.rect.x + space_;
      GLint  y = item.rect.y + space_;

      if ( i == focus_ )
      {
        gtx.setColor ( Color::white() );
      }
      else
      {
        gtx.setColor ( Color::black() );
      }

      txp.begin     ( gtx );
      txp.setCursor ( x, y );
      txp.writeLine ( gtx, item.label );
      txp.end       ( gtx );
    }
  }
}


//-----------------------------------------------------------------------
//   setFocus_
//-----------------------------------------------------------------------


void MenuNode::setFocus_

  ( RenderContext&  rtx,
    GLint           xPos,
    GLint           yPos )

{
  if ( focus_ != FOCUS_NONE_ )
  {
    if ( items_[focus_].rect.contains( xPos, yPos ) )
    {
      return;
    }
  }

  idx_t  oldFocus = focus_;

  focus_ = FOCUS_NONE_;

  if ( rect_.contains( xPos, yPos ) )
  {
    for ( idx_t i = 0; i < items_.size(); i++ )
    {
      const Item_&  item = items_[i];

      if ( item.rect.contains( xPos, yPos ) )
      {
        if ( item.label.size() > 0 )
        {
          focus_ = i;
        }

        break;
      }
    }
  }

  if ( focus_ != oldFocus )
  {
    rtx.postFastRedraw ( this );
  }
}


//-----------------------------------------------------------------------
//   moveFocus_
//-----------------------------------------------------------------------


void MenuNode::moveFocus_

  ( RenderContext&  rtx,
    int             dir )

{
  if ( focus_ == FOCUS_NONE_ )
  {
    return;
  }

  idx_t  i = focus_ + dir;

  while ( i >= 0 && i < items_.size() )
  {
    if ( items_[i].label.size() > 0 )
    {
      focus_ = i;

      rtx.postFastRedraw ( this );

      break;
    }

    i += dir;
  }
}


//-----------------------------------------------------------------------
//   selectFocus_
//-----------------------------------------------------------------------


void MenuNode::selectFocus_ ( RenderContext& rtx )
{
  const idx_t  i = focus_;

  hide ( rtx );

  if ( focus_ != FOCUS_NONE_ )
  {
    selectEvent.emit ( rtx, items_[i].label );
  }
}


JEM_END_PACKAGE( gl )
