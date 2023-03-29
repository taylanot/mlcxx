
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

#ifndef JEM_GL_TEXTPROJECTOR_H
#define JEM_GL_TEXTPROJECTOR_H

#include <jem/gl/Rect.h>
#include <jem/gl/Transform.h>
#include <jem/gl/TextPainter.h>


JEM_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   class TextProjector
//-----------------------------------------------------------------------


class TextProjector : public TextPainter
{
 public:

  typedef TextPainter       Super;


  explicit                  TextProjector

    ( const Font&             font   = Font (),
      Alignment               hAlign = LEFT_ALIGNED,
      Alignment               vAlign = BOT_ALIGNED );

  void                      begin

    ( GLContext&              gtx );

  void                      begin

    ( GLContext&              gtx,
      const Transform&        proj );

  void                      end

    ( GLContext&              gtx );

  inline void               setCursor

    ( GLdouble                x,
      GLdouble                y );

  void                      setCursor

    ( GLdouble                x,
      GLdouble                y,
      GLdouble                z );

  inline bool               isVisible   () const noexcept;


 private:

  Transform                 proj_;
  Rect                      vport_;
  bool                      active_;

};


//-----------------------------------------------------------------------
//   class OrthoTextScope
//-----------------------------------------------------------------------


class OrthoTextScope
{
 public:

  explicit inline           OrthoTextScope

    ( GLContext&              gtx,
      TextProjector&          txp );

  inline                   ~OrthoTextScope  ();


 private:

  TextProjector&            txp_;
  GLContext&                gtx_;

};




//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class TextProjector
//=======================================================================

//-----------------------------------------------------------------------
//   setCursor
//-----------------------------------------------------------------------


inline void TextProjector::setCursor

  ( GLdouble  x,
    GLdouble  y )

{
  setCursor ( x, y, 0.0 );
}


//-----------------------------------------------------------------------
//   isVisible
//-----------------------------------------------------------------------


inline bool TextProjector::isVisible () const noexcept
{
  return (vport_.contains( xPos, yPos ) &&
          zPos >= 0.0 && zPos <= 1.0);
}


//=======================================================================
//   class OrthoTextScope
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


inline OrthoTextScope::OrthoTextScope

  ( GLContext&      gtx,
    TextProjector&  txp ) :

    txp_ ( txp ),
    gtx_ ( gtx )

{
  txp_.begin ( gtx );
}


inline OrthoTextScope::~OrthoTextScope ()
{
  txp_.end ( gtx_ );
}


JEM_END_PACKAGE( gl )

#endif
