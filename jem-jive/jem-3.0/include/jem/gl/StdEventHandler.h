
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

#ifndef JEM_GL_STDEVENTHANDLER_H
#define JEM_GL_STDEVENTHANDLER_H

#include <jem/gl/Color.h>
#include <jem/gl/EventHandler.h>


JEM_BEGIN_PACKAGE( gl )


class Camera;


//-----------------------------------------------------------------------
//   class StdEventHandler
//-----------------------------------------------------------------------


class StdEventHandler : public EventHandler
{
 public:

  JEM_DECLARE_CLASS       ( StdEventHandler, EventHandler );

  static const char         NULL_MODE;
  static const char         ZOOM_MODE;
  static const char         ROTATION_MODE;
  static const char         TRANSLATION_MODE;


                            StdEventHandler

    ( const Ref<Camera>&      camera,
      const Ref<GfxNode>&     world );

  virtual void              reset             ()       override;

  virtual void              redraw

    ( RenderContext&          rtx,
      int                     action,
      const Properties&       params )                 override;

  virtual void              newView

    ( RenderContext&          rtx,
      const Rect&             vport )                  override;

  virtual void              keyPressed

    ( RenderContext&          rtx,
      const KeyPress&         key )                    override;

  virtual void              mouseDragged

    ( RenderContext&          rtx,
      const MouseDrag&        mouse )                  override;

  virtual bool              exeCommand

    ( RenderContext&          rtx,
      const String&           cmd )                    override;

  virtual void              printHelp

    ( PrintWriter&            pr )                     override;

  virtual void              initMenu

    ( Menu&                   menu )             const override;

  virtual void              listOptions

    ( Menu&                   menu )             const override;

  virtual void              setFocus

    ( const Ref<GfxNode>&     node )                   override;

  inline char               getMode           () const noexcept;

  void                      setMode

    ( char                    mode );


 public:

  Color                     focusColor;


 protected:

  virtual                  ~StdEventHandler   ();

  void                      zoomView_

    ( RenderContext&          rtx,
      int                     step );

  void                      scaleFocus_

    ( RenderContext&          rtx,
      int                     step );


 protected:

  Ref<Camera>               camera_;
  Ref<GfxNode>              world_;
  Ref<GfxNode>              focus_;
  Ref<GfxNode>              noFocus_;
  char                      mode_;


 private:

  void                      centerObj_        ();

};



//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   getMode
//-----------------------------------------------------------------------


inline char StdEventHandler::getMode () const noexcept
{
  return mode_;
}


JEM_END_PACKAGE( gl )

#endif
