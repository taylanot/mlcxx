
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

#ifndef JEM_GL_NATIVEWINDOW_H
#define JEM_GL_NATIVEWINDOW_H

#include <jem/base/Object.h>
#include <jem/base/Monitor.h>
#include <jem/util/Timer.h>
#include <jem/gl/import.h>
#include <jem/gl/GLContext.h>
#include <jem/gl/EventQueue.h>
#include <jem/gl/CallbackQueue.h>
#include <jem/gl/RenderContext.h>


JEM_BEGIN_PACKAGE( gl )


class GfxError;
class ErrorHandler;
class NativeDisplay;

using jem::util::Timer;


//-----------------------------------------------------------------------
//   class NativeWindow
//-----------------------------------------------------------------------


class NativeWindow : public Object,
                     public RenderContext
{
 public:

  typedef Object            Super;
  typedef NativeWindow      Self;
  typedef NativeDisplay     Display;


  virtual void              close         ();
  virtual void              setCurrent    () = 0;
  virtual bool              shouldClose   () const;

  virtual void              postCallback

    ( Callback                cb,
      const Ref<Object>&      obj,
      const Time&             tm )                 override;

  virtual void              repostCallback

    ( Callback                cb,
      const Ref<Object>&      obj,
      const Time&             tm )                 override;

  virtual void              cancelCallback

    ( Callback                cb,
      const Ref<Object>&      obj )                override;

  virtual void              cancelCallbacks

    ( Callback                cb )                 override;

  virtual GLContext&        getGLContext  ()       override;
  virtual Vertex2i          getMousePos   () const override;
  virtual int               getActionID   () const override;
  int                       incrActionID  ();
  void                      startWork     ();
  void                      stopWork      ();
  double                    getWorkLoad   () const;

  void                      contextError

    ( const String&           where )        const;


 public:

  Monitor                   mon;
  EventQueue                events;
  CallbackQueue             callbacks;
  Ref<ErrorHandler>         errHandler;

  Ref<GfxError>             err;
  Ref<GfxNode>              scene;
  Vertex2i                  mousePos;


 protected:

                            NativeWindow  ();
  virtual                  ~NativeWindow  ();

  virtual void              postRedraw_

    ( const Ref<GfxNode>&     root,
      bool                    fast )               override;


 private:

  GLContext                 gtx_;
  int                       actionID_;

  Timer                     runTimer_;
  Timer                     workTimer_;
  double                    workLoad_;
  double                    workLoad0_;

};


JEM_END_PACKAGE( gl )

#endif
