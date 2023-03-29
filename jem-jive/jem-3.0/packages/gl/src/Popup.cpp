
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


#include <jem/base/Time.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/RuntimeException.h>
#include <jem/util/Properties.h>
#include <jem/gl/Actions.h>
#include <jem/gl/Viewer.h>
#include <jem/gl/Compound.h>
#include <jem/gl/RenderContext.h>
#include <jem/gl/Popup.h>


JEM_DEFINE_CLASS( jem::gl::Popup );


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class Popup
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  Popup::HIDE_ = "_HIDE_POPUP";


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


Popup::Popup ( const String& name ) : Super ( name )
{
  actionID_ = -1;
}


Popup::~Popup ()
{}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool Popup::takeAction

  ( const String&      action,
    const Properties&  params )

{
  if ( action == Actions::GIVE_FOCUS )
  {
    hide ();

    return true;
  }

  if ( action == HIDE_ )
  {
    hide ();

    return true;
  }

  if ( action == Actions::SET_FONT )
  {
    font = Font::getDefault ();

    return true;
  }

  return false;
}


//-----------------------------------------------------------------------
//   takeGLAction
//-----------------------------------------------------------------------


bool Popup::takeGLAction

  ( RenderContext&     rtx,
    int                action,
    const Properties&  params )

{
  bool  result = false;
  bool  down;


  switch ( action )
  {
  case GLActions::REDRAW:
  case GLActions::REDRAW_FAST:

    if ( actionID_ < 0 )
    {
      rtx.cancelCallback ( hideCallback_, this );
      hide               ();
    }
    else
    {
      redraw ( rtx );
    }

    result = true;

    break;

  case GLActions::MOUSE_EVENT:

    params.get ( down, GLActionParams::MOUSE_DOWN );

    if ( ! down )
    {
      break;
    }

  case GLActions::INIT_VIEW:
  case GLActions::KEY_EVENT:

    if ( (actionID_ >= 0) && (rtx.getActionID() != actionID_) )
    {
      rtx.repostCallback ( & hideCallback_, this,
                           Time::now() + 0.4_sec );

      actionID_ = -1;
      result    =  true;
    }

    break;
  }

  return result;
}


//-----------------------------------------------------------------------
//   hide
//-----------------------------------------------------------------------


void Popup::hide ()
{
  GfxNode*   parent = getParent ();

  if ( parent )
  {
    Compound*  cmp = dynamic_cast<Compound*> ( parent );

    if ( ! cmp )
    {
      String  parentName = parent->getName ();

      throw RuntimeException (
        getContext (),
        String::format (
          "failed to hide this message box because its parent "
          "node `%s\' is not a compound node",
          parentName
        )
      );
    }

    cmp->eraseChild ( this );
  }

  actionID_ = -1;
}


//-----------------------------------------------------------------------
//   hideAt
//-----------------------------------------------------------------------


void Popup::hideAt

  ( RenderContext&  rtx,
    const Time&     tm )

{
  if ( getParent() )
  {
    rtx.repostCallback ( & hideCallback_, this, tm );
  }
}


//-----------------------------------------------------------------------
//   show
//-----------------------------------------------------------------------


void Popup::show

  ( Viewer&              viewer,
    const Ref<GfxNode>&  node )

{
  show_                 ( node.get() );
  viewer.postFastRedraw ( node );

  actionID_ = 0;
}


void Popup::show

  ( RenderContext&       rtx,
    const Ref<GfxNode>&  node )

{
  show_              ( node.get() );
  rtx.postFastRedraw ( node );

  actionID_ = rtx.getActionID ();
}


//-----------------------------------------------------------------------
//   show_
//-----------------------------------------------------------------------


void Popup::show_ ( GfxNode* node )
{
  GfxNode*   next;
  Compound*  comp;


  if ( getParent() )
  {
    return;
  }

  comp = dynamic_cast<Compound*> ( node );

  if ( ! node )
  {
    return;
  }

  next = node->getParent ();

  while ( next )
  {
    node = next;
    next = node->getParent ();

    if ( ! comp )
    {
      comp = dynamic_cast<Compound*> ( node );
    }
  }

  if ( node )
  {
    node->takeAction ( HIDE_, Properties() );
  }

  if ( comp )
  {
    comp->addChild ( this );
  }
}


//-----------------------------------------------------------------------
//   hideCallback_
//-----------------------------------------------------------------------


void Popup::hideCallback_

  ( RenderContext&      rtx,
    const Ref<Object>&  obj )

{
  Ref<Self>  self = staticCast<Self> ( obj );

  if ( self->getParent() )
  {
    rtx.postFastRedraw ( self->getParent() );
    self->hide         ();
  }
}


JEM_END_PACKAGE( gl )
