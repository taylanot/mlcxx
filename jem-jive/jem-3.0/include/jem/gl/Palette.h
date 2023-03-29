
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

#ifndef JEM_GL_PALETTE_H
#define JEM_GL_PALETTE_H

#include <jem/base/Object.h>
#include <jem/util/event/Event.h>
#include <jem/gl/import.h>
#include <jem/gl/Color.h>
#include <jem/gl/Alignment.h>


JEM_BEGIN_PACKAGE( gl )


class Rect;
class ColorMap;
class GLContext;
class TextPainter;


//-----------------------------------------------------------------------
//   class Palette
//-----------------------------------------------------------------------


class Palette : public Object
{
 public:

  JEM_DECLARE_CLASS       ( Palette, Object );


  util::Event< Self& >      changeEvent;


  virtual void              getColor

    ( Color&                  col,
      double                  val )          const = 0;

  virtual double            findValue

    ( double&                 val,
      double                  prec,
      const Color&            col )          const;

  virtual double            getMinValue   () const = 0;
  virtual double            getMaxValue   () const = 0;
  virtual GLfloat           getMinAlpha   () const = 0;

  virtual void              makeColorMap

    ( ColorMap&               cmap )         const;

  virtual void              configure

    ( const Properties&       props );

  virtual void              getConfig

    ( const Properties&       props )        const;

  virtual void              listProps

    ( const Properties&       props )        const;

  void                      render

    ( GLContext&              gtx,
      const Rect&             rect,
      const String&           title,
      Alignment               align,
      double                  scale = 1.0,
      double                  bias  = 0.0 )  const;

  virtual void              fillRectangle

    ( GLContext&              gtx,
      GLint                   width,
      GLint                   height )       const;

  virtual void              drawTickMarks

    ( GLContext&              gtx,
      TextPainter&            txp,
      GLint                   width,
      GLint                   height,
      GLint                   xtick,
      Alignment               align,
      double                  scale,
      double                  bias )         const;


 protected:

  virtual                  ~Palette       ();


 private:

  double                    findValue_

    ( double&                 val,
      double                  prec,
      const Color&            col,
      int                     n )            const;

  void                      drawPaletteV1_

    ( GLContext&              gtx,
      TextPainter&            txp,
      GLint                   width,
      GLint                   height,
      GLint                   xtick,
      Alignment               align,
      double                  scale,
      double                  bias )         const;

  void                      drawPaletteV3_

    ( GLContext&              gtx,
      TextPainter&            txp,
      GLint                   width,
      GLint                   height,
      GLint                   xtick,
      Alignment               align,
      double                  scale,
      double                  bias )         const;

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


double                      mapColorToValue

  ( double&                   val,
    double                    prec,
    const Color&              color,
    const Color&              minColor,
    const Color&              maxColor );

void                        makeLinearColorMap

  ( ColorMap&                 colorMap,
    idx_t                     colorCount,
    const Palette&            palette );


JEM_END_PACKAGE( gl )

#endif
