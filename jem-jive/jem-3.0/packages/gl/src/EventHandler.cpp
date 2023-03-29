
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
#include <jem/gl/EventHandler.h>


JEM_DEFINE_CLASS( jem::gl::EventHandler );


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class EventHandler
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


EventHandler::~EventHandler ()
{}


//-----------------------------------------------------------------------
//   reset
//-----------------------------------------------------------------------


void EventHandler::reset ()
{}


//-----------------------------------------------------------------------
//   redraw
//-----------------------------------------------------------------------


void EventHandler::redraw

  ( RenderContext&     rtx,
    int                action,
    const Properties&  params )

{}


//-----------------------------------------------------------------------
//   newView
//-----------------------------------------------------------------------


void EventHandler::newView

  ( RenderContext&  rtx,
    const Rect&     vport )

{}


//-----------------------------------------------------------------------
//   keyPressed
//-----------------------------------------------------------------------


void EventHandler::keyPressed

  ( RenderContext&   rtx,
    const KeyPress&  key )

{}


//-----------------------------------------------------------------------
//   mouseClicked
//-----------------------------------------------------------------------


void EventHandler::mouseClicked

  ( RenderContext&     rtx,
    const MouseClick&  mouse )

{}


//-----------------------------------------------------------------------
//   mouseDragged
//-----------------------------------------------------------------------


void EventHandler::mouseDragged

  ( RenderContext&    rtx,
    const MouseDrag&  mouse )

{}


//-----------------------------------------------------------------------
//   exeCommand
//-----------------------------------------------------------------------


bool EventHandler::exeCommand

  ( RenderContext&  rtx,
    const String&   cmd )

{
  return false;
}


//-----------------------------------------------------------------------
//   printHelp
//-----------------------------------------------------------------------


void EventHandler::printHelp ( PrintWriter& pr )
{}


//-----------------------------------------------------------------------
//   initMenu
//-----------------------------------------------------------------------


void EventHandler::initMenu ( Menu& menu ) const
{}


//-----------------------------------------------------------------------
//   listOptions
//-----------------------------------------------------------------------


void EventHandler::listOptions ( Menu& menu ) const
{}


//-----------------------------------------------------------------------
//   setFocus
//-----------------------------------------------------------------------


void EventHandler::setFocus ( const Ref<GfxNode>& node )
{}


JEM_END_PACKAGE( gl )
