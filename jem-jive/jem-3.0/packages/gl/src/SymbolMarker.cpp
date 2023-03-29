
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
#include <jem/util/Properties.h>
#include <jem/gl/GLContext.h>
#include <jem/gl/GLFunctions1.h>
#include <jem/gl/PropertyNames.h>
#include <jem/gl/Symbols.h>
#include <jem/gl/RenderContext.h>
#include <jem/gl/SymbolMarker.h>


JEM_DEFINE_CLASS( jem::gl::SymbolMarker );


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class SymbolMarker
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


SymbolMarker::SymbolMarker

  ( char          sym,
    const Color&  color ) :

    writer_ ( Font::getDefault( SYMBOL_FONT ), CENTERED, CENTERED )

{
  symbol = sym;
  color_ = clamped ( color );
}


SymbolMarker::~SymbolMarker ()
{}


//-----------------------------------------------------------------------
//   toString
//-----------------------------------------------------------------------


String SymbolMarker::toString () const
{
  if ( ! symbol )
  {
    return "none";
  }
  else
  {
    String  name = Symbols::getName ( symbol );

    if ( ! name.size() )
    {
      name = "symbol";
    }

    return name;
  }
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void SymbolMarker::configure ( const Properties& props )
{
  String  sizeName;

  if ( props.find( sizeName, PropNames::SIZE ) )
  {
    FontSize  fnSize = SMALL_FONT;

    sizeName = sizeName.toLower ();

    if      ( sizeName == "small" )
    {
      fnSize = SMALL_FONT;
    }
    else if ( sizeName == "medium" )
    {
      fnSize = MEDIUM_FONT;
    }
    else if ( sizeName == "large" )
    {
      fnSize = LARGE_FONT;
    }
    else
    {
      props.propertyError (
        PropNames::SIZE,
        String::format (
          "invalid size: %s; "
          "should be `small\', `medium\' or `large\'",
          sizeName
        )
      );
    }

    writer_.font = Font ( SYMBOL_FONT, fnSize );
  }

  color_.configure ( PropNames::COLOR, props );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void SymbolMarker::getConfig ( const Properties& conf ) const
{
  String  sizeName;

  switch ( writer_.font.getSize() )
  {
  case SMALL_FONT:

    sizeName = "small";
    break;

  case MEDIUM_FONT:

    sizeName = "medium";
    break;

  case LARGE_FONT:

    sizeName = "large";
    break;
  }

  conf  .set       ( PropNames::SIZE,  sizeName );
  color_.getConfig ( PropNames::COLOR, conf );
}


//-----------------------------------------------------------------------
//   listProps
//-----------------------------------------------------------------------


void SymbolMarker::listProps ( const Properties& conf ) const
{
  conf.set ( PropNames::SIZE,
             "The size of the symbol markers" );
  conf.set ( PropNames::COLOR,
             "The color of the markers" );
}


//-----------------------------------------------------------------------
//   setFont
//-----------------------------------------------------------------------


void SymbolMarker::setFont ()
{
  writer_.font = Font::getDefault ( SYMBOL_FONT );
}


//-----------------------------------------------------------------------
//   beginDraw
//-----------------------------------------------------------------------


void SymbolMarker::beginDraw

  ( GLContext&  gtx,
    const Box&  box,
    idx_t       count )

{
  if ( symbol )
  {
    gtx.pushAttribs ( GLContext::COLOR | GLContext::LIGHTING );
    gtx.setLighting ( false );
    gtx.setColor    ( color_ );
    writer_.begin   ( gtx );
  }
}


//-----------------------------------------------------------------------
//   endDraw
//-----------------------------------------------------------------------


void SymbolMarker::endDraw ( GLContext& gtx )
{
  if ( symbol )
  {
    writer_.end    ( gtx );
    gtx.popAttribs ();
  }
}


//-----------------------------------------------------------------------
//   drawMarker
//-----------------------------------------------------------------------


void SymbolMarker::drawMarker

  ( GLContext&      gtx,
    idx_t           tag,
    const GLfloat*  pos )

{
  if ( symbol )
  {
    writer_.setCursor ( pos[0], pos[1], pos[2] );
    writer_.write     ( gtx, symbol );
  }
}


//-----------------------------------------------------------------------
//   setColor
//-----------------------------------------------------------------------


void SymbolMarker::setColor ( const Color& color )
{
  color_ = color;

  clamp ( color_ );
}


//-----------------------------------------------------------------------
//   getColor
//-----------------------------------------------------------------------


Color SymbolMarker::getColor () const
{
  return color_;
}


JEM_END_PACKAGE( gl )
