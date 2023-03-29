
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

#ifndef JEM_GL_CONTROLLER_H
#define JEM_GL_CONTROLLER_H

#include <jem/base/Object.h>
#include <jem/util/Properties.h>
#include <jem/util/event/Event.h>
#include <jem/gl/import.h>


JEM_BEGIN_PACKAGE( gl )


class Menu;
class Viewer;
class Camera;
class GfxNode;
class MenuNode;
class MultiNode;
class EventNode;
class EventHandler;
class KeyPress;
class MouseClick;
class FrameGrabber;
class RenderContext;


//-----------------------------------------------------------------------
//   class Controller
//-----------------------------------------------------------------------


class Controller : public Object
{
 public:

  JEM_DECLARE_CLASS       ( Controller, Object );

  typedef util::Event
    < Self&,
      RenderContext& >      Event;

  Event                     helpEvent;
  Event                     exitEvent;
  Event                     resetEvent;
  Event                     config0Event;


                            Controller

    ( Ref<MultiNode>          world,
      Ref<EventHandler>       handler );

  void                      setSnapFile

    ( const String&           name );

  void                      setSnapFile

    ( const String&           name,
      Ref<FrameGrabber>       grabber );

  String                    makeSnapshot

    ( Viewer&                 viewer );

  void                      printHelp

    ( PrintWriter&            pr )           const;

  void                      reset         ();

  void                      setConfig0

    ( const Properties&       props );

  Properties                getConfig0    () const;
  EventNode*                getEventNode  () const;


 protected:

  virtual                  ~Controller    ();


 private:

  void                      redraw_

    ( RenderContext&          rtx,
      int                     action,
      const Properties&       params );

  void                      newFocus_

    ( bool                    state );

  void                      drawNames_

    ( RenderContext&          rtx );

  void                      keyPressed_

    ( RenderContext&          rtx,
      const KeyPress&         key );

  void                      mouseClicked_

    ( RenderContext&          rtx,
      const MouseClick&       mouse );

  void                      exeCommand_

    ( RenderContext&          rtx,
      const String&           cmd );

  void                      showHelp_     ();

  void                      showMenu_

    ( RenderContext&          rtx );

  void                      showOptions_

    ( RenderContext&          rtx );

  void                      showNodes_

    ( RenderContext&          rtx );

  void                      nodeErased_

    ( const Ref<GfxNode>&     node );

  void                      selectNode_

    ( RenderContext&          rtx,
      const String&           name );

  void                      selectNext_   ();
  void                      selectPrev_   ();

  void                      makeSnapshot_

    ( RenderContext&          rtx );

  void                      initSnapshot_

    ( String&                 fname,
      Ref<OutputStream>&      output )       const;

  String                    getSnapFile_  () const;

  void                      saveConfig0_

    ( RenderContext&          rtx );


 private:

  Ref<MultiNode>            world_;
  Ref<EventNode>            evNode_;
  Ref<EventHandler>         evHandler_;

  Ref<FrameGrabber>         grabber_;
  String                    snapFile_;
  idx_t                     snapCount_;

  Ref<GfxNode>              focus_;
  Properties                config0_;
  bool                      showNames_;

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


Ref<Controller>             newController2D

  ( const Ref<Camera>&        camera,
    const Ref<MultiNode>&     world );

Ref<Controller>             newController3D

  ( const Ref<Camera>&        camera,
    const Ref<MultiNode>&     world );


JEM_END_PACKAGE( gl )

#endif
