
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

#ifndef JEM_GL_VIEWER_H
#define JEM_GL_VIEWER_H

#include <jem/base/Object.h>
#include <jem/gl/import.h>


JEM_BEGIN_PACKAGE( gl )


class GfxNode;
class FrameGrabber;


//-----------------------------------------------------------------------
//   class Viewer
//-----------------------------------------------------------------------


class Viewer : public Object
{
 public:

  JEM_DECLARE_CLASS       ( Viewer, Object );


  virtual void              freeze          ()       = 0;
  virtual void              resume          ()       = 0;
  virtual bool              shouldClose     () const = 0;

  virtual void              setScene

    ( const Ref<GfxNode>&     scene )                = 0;

  virtual Ref<GfxNode>      getScene        () const = 0;
  virtual double            getWorkLoad     () const = 0;
  virtual void              flushEvents     ()       = 0;

  void                      postRedraw      ();

  void                      postRedraw

    ( const Ref<GfxNode>&     node );

  void                      postFastRedraw  ();

  void                      postFastRedraw

    ( const Ref<GfxNode>&     node );

  void                      grabFrame

    ( OutputStream&           output,
      FrameGrabber&           grabber );

  void                      grabFrame

    ( const Ref<GfxNode>&     node,
      OutputStream&           output,
      FrameGrabber&           grabber );


 protected:

  virtual                  ~Viewer          ();

  virtual void              postRedraw_

    ( const Ref<GfxNode>&     node )                 = 0;

  virtual void              postFastRedraw_

    ( const Ref<GfxNode>&     node )                 = 0;

  virtual void              grabFrame_

    ( const Ref<GfxNode>&     node,
      OutputStream&           output,
      FrameGrabber&           grabber )              = 0;

};


JEM_END_PACKAGE( gl )

#endif
