
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


#include <jem/base/assert.h>
#include <jem/base/RuntimeException.h>
#include <jem/io/OutputStream.h>
#include <jem/gl/DummyNode.h>
#include <jem/gl/FrameGrabber.h>
#include "NativeDisplay.h"
#include "NativeViewer.h"


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class NativeViewer
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


NativeViewer::NativeViewer

  ( const Ref<Window>& win ) :

    window_  ( win ),
    display_ ( NativeDisplay::getInstance() )

{
  JEM_PRECHECK ( win );

  scene_ = newInstance<DummyNode> ();
}


NativeViewer::~NativeViewer ()
{
  display_->closeWindow ( window_ );
}


//-----------------------------------------------------------------------
//   freeze
//-----------------------------------------------------------------------


void NativeViewer::freeze ()
{
  display_->freezeWindow ( window_ );
}


//-----------------------------------------------------------------------
//   resume
//-----------------------------------------------------------------------


void NativeViewer::resume ()
{
  display_->resumeWindow ( window_, scene_ );
}


//-----------------------------------------------------------------------
//   shouldClose
//-----------------------------------------------------------------------


bool NativeViewer::shouldClose () const
{
  return display_->shouldClose ( window_ );
}


//-----------------------------------------------------------------------
//   setScene
//-----------------------------------------------------------------------


void NativeViewer::setScene ( const Ref<GfxNode>& scene )
{
  JEM_PRECHECK ( scene );

  scene_ = scene;

  display_->setScene ( window_, scene );
}


//-----------------------------------------------------------------------
//   getScene
//-----------------------------------------------------------------------


Ref<GfxNode> NativeViewer::getScene () const
{
  return scene_;
}


//-----------------------------------------------------------------------
//   getWorkLoad
//-----------------------------------------------------------------------


double NativeViewer::getWorkLoad () const
{
  return window_->getWorkLoad ();
}


//-----------------------------------------------------------------------
//   flushEvents
//-----------------------------------------------------------------------


void NativeViewer::flushEvents ()
{
  display_->flushEvents ( window_, scene_ );
}


//-----------------------------------------------------------------------
//   postRedraw_
//-----------------------------------------------------------------------


void NativeViewer::postRedraw_ ( const Ref<GfxNode>& scene )
{
  RedrawEventPtr  ev = new RedrawEvent ( scene );

  ev->flags |= RedrawEvent::EXTERNAL;

  window_->mon.lock    ();
  window_->events.push ( ev.release() );
  window_->mon.unlock  ();
}


//-----------------------------------------------------------------------
//   postFastRedraw_
//-----------------------------------------------------------------------


void NativeViewer::postFastRedraw_ ( const Ref<GfxNode>& scene )
{
  RedrawEventPtr  ev = new RedrawEvent ( scene );

  ev->flags |= RedrawEvent::FAST;
  ev->flags |= RedrawEvent::EXTERNAL;

  window_->mon.lock    ();
  window_->events.push ( ev.release() );
  window_->mon.unlock  ();
}


//-----------------------------------------------------------------------
//   grabFrame_
//-----------------------------------------------------------------------


void NativeViewer::grabFrame_

  ( const Ref<GfxNode>&  node,
    OutputStream&        output,
    FrameGrabber&        grabber )

{
  display_->grabFrame ( window_, & output, & grabber, node );
}


JEM_END_PACKAGE( gl )
