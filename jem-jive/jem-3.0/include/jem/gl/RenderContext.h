
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

#ifndef JEM_GL_RENDERCONTEXT_H
#define JEM_GL_RENDERCONTEXT_H

#include <jem/base/Object.h>
#include <jem/gl/Size.h>
#include <jem/gl/Vertex.h>


namespace jem
{
  class Time;
}


JEM_BEGIN_PACKAGE( gl )


class GfxNode;
class GLContext;


//-----------------------------------------------------------------------
//   class RenderContext
//-----------------------------------------------------------------------


class RenderContext
{
 public:

  typedef void            (*Callback)

    ( RenderContext&          rtx,
      const Ref<Object>&      obj );


  virtual GLContext&        getGLContext    ()       = 0;
  virtual Vertex2i          getMousePos     () const = 0;
  virtual Size2i            getWindowSize   () const = 0;
  virtual int               getActionID     () const = 0;

  void                      postRedraw

    ( const Ref<GfxNode>&     node );

  void                      postFastRedraw

    ( const Ref<GfxNode>&     node );

  virtual void              postCallback

    ( Callback                cb,
      const Ref<Object>&      obj,
      const Time&             tm )                   = 0;

  virtual void              repostCallback

    ( Callback                cb,
      const Ref<Object>&      obj,
      const Time&             tm )                   = 0;

  virtual void              cancelCallback

    ( Callback                cb,
      const Ref<Object>&      obj )                  = 0;

  virtual void              cancelCallbacks

    ( Callback                cb )                   = 0;

  virtual void              swapBuffers     ()       = 0;


 protected:

                            RenderContext   ();
  virtual                  ~RenderContext   ();

  virtual void              postRedraw_

    ( const Ref<GfxNode>&     root,
      bool                    fast )                 = 0;


 private:

                            RenderContext

    ( const RenderContext&    rhs );

  RenderContext&            operator =

    ( const RenderContext&    rhs );

};


JEM_END_PACKAGE( gl )

#endif
