
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jive, an object oriented toolkit for solving
 *  partial differential equations.
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
 *  This file is part of Jive, an object oriented toolkit for
 *  solving partial differential equations.
 *
 *  Jive version: 3.0
 *  Date:         Fri 20 Dec 14:30:12 CET 2019
 */


#include <jem/base/assert.h>
#include <jem/base/Once.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalInputException.h>
#include <jem/util/None.h>
#include <jem/util/Flex.h>
#include <jem/util/Properties.h>
#include <jem/util/HashDictionary.h>
#include <jem/gl/Symbols.h>
#include <jem/gl/PointMarker.h>
#include <jem/gl/OrbMarker.h>
#include <jem/gl/TeeMarker.h>
#include <jem/gl/NumberMarker.h>
#include <jem/gl/SymbolMarker.h>
#include <jive/util/CtorMap.h>
#include <jive/util/Factory.h>
#include <jive/gl/Viewdat.h>
#include <jive/gl/MarkerStyleTable.h>


JEM_DEFINE_CLASS( jive::gl::MarkerStyleTable );


JIVE_BEGIN_PACKAGE( gl )


using jem::staticCast;
using jem::newInstance;
using jem::gl::Symbols;
using jem::gl::SymbolMarker;
using jem::gl::MarkerStyle;


//=======================================================================
//   MarkerStyle constructor funcs
//=======================================================================


static Ref<MarkerStyle>  newPointMarker ()
{
  return newInstance<jem::gl::PointMarker> ();
}


static Ref<MarkerStyle>  newOrbMarker ()
{
  return newInstance<jem::gl::OrbMarker> ();
}


static Ref<MarkerStyle>  newTeeMarker ()
{
  return newInstance<jem::gl::TeeMarker> ();
}


static Ref<MarkerStyle>  newNumberMarker ()
{
  return newInstance<jem::gl::NumberMarker> ();
}


static Ref<MarkerStyle>  newCircleMarker ()
{
  return newInstance<SymbolMarker> ( Symbols::CIRCLE );
}


static Ref<MarkerStyle>  newDiamondMarker ()
{
  return newInstance<SymbolMarker> ( Symbols::DIAMOND );
}


static Ref<MarkerStyle>  newDiscMarker ()
{
  return newInstance<SymbolMarker> ( Symbols::DISC );
}


static Ref<MarkerStyle>  newSquareMarker ()
{
  return newInstance<SymbolMarker> ( Symbols::SQUARE );
}



//=======================================================================
//   class MarkerStyleTable::CtorMap_
//=======================================================================


class MarkerStyleTable::CtorMap_ : public util::CtorMap<StyleCtor>
{};


//=======================================================================
//   class MarkerStyleTable
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


MarkerStyleTable::MarkerStyleTable ()
{
  static jem::Once  once = JEM_ONCE_INITIALIZER;

  jem::runOnce ( once, initOnce_ );

  styles_ = newInstance<jem::util::HashDict> ();

  // Add default styles

  styles_->insert ( "points",   newPointMarker   () );
  styles_->insert ( "orbs",     newOrbMarker     () );
  styles_->insert ( "numbers",  newNumberMarker  () );
  styles_->insert ( "circles",  newCircleMarker  () );
  styles_->insert ( "diamonds", newDiamondMarker () );
  styles_->insert ( "discs",    newDiscMarker    () );
  styles_->insert ( "squares",  newSquareMarker  () );
}


MarkerStyleTable::~MarkerStyleTable ()
{}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void MarkerStyleTable::configure ( const Properties& props )
{
  using jem::util::isNone;
  using jem::util::Flex;
  using jem::util::DictEnum;

  Ref<DictEnum>  e;
  Ref<Style>     s;
  Ref<Object>    obj;

  Flex<String>   hitList;
  String         name;


  for ( e = styles_->enumerate(); ! e->atEnd(); e->toNext() )
  {
    name = e->getKey   ();

    if ( ! props.find( obj, name ) )
    {
      continue;
    }

    if ( isNone( obj ) )
    {
      hitList.pushBack ( std::move( name ) );
      continue;
    }

    s = staticCast<Style> ( e->getValue() );

    s->configure ( props.getProps( name ) );
  }

  if ( hitList.size() )
  {
    e = nullptr;

    for ( idx_t i = 0; i < hitList.size(); i++ )
    {
      styles_->erase ( hitList[i] );
    }
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void MarkerStyleTable::getConfig ( const Properties& props ) const
{
  using jem::util::DictEnum;

  Ref<DictEnum>  e;
  Ref<Style>     s;

  String         name;


  for ( e = styles_->enumerate(); ! e->atEnd(); e->toNext() )
  {
    name = e->getKey ();
    s    = staticCast<Style> ( e->getValue() );

    s->getConfig ( props.makeProps( name ) );
  }
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void MarkerStyleTable::clear ()
{
  styles_->clear ();
}


//-----------------------------------------------------------------------
//   addStyle
//-----------------------------------------------------------------------


void MarkerStyleTable::addStyle

  ( const String&      name,
    const Ref<Style>&  style )

{
  JEM_PRECHECK ( style );

  styles_->insert ( name, style );
}


//-----------------------------------------------------------------------
//   addStyles
//-----------------------------------------------------------------------


void MarkerStyleTable::addStyles

  ( const StringVector&  names,
    const Properties&    conf,
    const Properties&    props )

{
  const idx_t   n = names.size ();

  Ref<Style>  s;


  for ( idx_t i = 0; i < n; i++ )
  {
    Properties  styleConf  = conf .makeProps ( names[i] );
    Properties  styleProps = props.getProps  ( names[i] );

    s = newStyle ( styleConf, styleProps );

    s->configure ( styleProps );
    s->getConfig ( styleConf  );

    styles_->insert ( names[i], s );
  }
}


//-----------------------------------------------------------------------
//   findStyle
//-----------------------------------------------------------------------


Ref<MarkerStyle> MarkerStyleTable::findStyle

  ( const String& name ) const

{
  return staticCast<Style> ( styles_->get( name ) );
}


//-----------------------------------------------------------------------
//   getStyle
//-----------------------------------------------------------------------


Ref<MarkerStyle> MarkerStyleTable::getStyle

  ( const String&  name,
    const String&  context ) const

{
  Ref<Style>  style = staticCast<Style> ( styles_->get( name ) );

  if ( ! style )
  {
    throw jem::IllegalInputException (
      context,
      String::format ( "undefined marker style: `%s\'", name )
    );
  }

  return style;
}


//-----------------------------------------------------------------------
//   listStyles
//-----------------------------------------------------------------------


StringVector MarkerStyleTable::listStyles () const
{
  using jem::util::DictEnum;

  Ref<DictEnum>  e;

  StringVector   names ( styles_->size() );

  idx_t          i = 0;


  for ( e = styles_->enumerate(); ! e->atEnd(); e->toNext() )
  {
    names[i++] = e->getKey ();
  }

  return names;
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


Ref<MarkerStyleTable> MarkerStyleTable::get

  ( const Properties& viewdat )

{
  Ref<Self>  self;

  viewdat.find ( self, Viewdat::MARKER_STYLES );

  if ( ! self )
  {
    self = newInstance<Self> ();

    viewdat.set ( Viewdat::MARKER_STYLES, self );
  }

  return self;
}


//-----------------------------------------------------------------------
//   declareStyle
//-----------------------------------------------------------------------


void MarkerStyleTable::declareStyle

  ( const String&  type,
    StyleCtor      ctor )

{
  JEM_PRECHECK ( ctor );

  CtorMap_::insert ( type, ctor );
}


//-----------------------------------------------------------------------
//   listStyleTypes
//-----------------------------------------------------------------------


StringVector MarkerStyleTable::listStyleTypes ()
{
  return CtorMap_::listKnownTypes ();
}


//-----------------------------------------------------------------------
//   newStyle
//-----------------------------------------------------------------------


Ref<MarkerStyle> MarkerStyleTable::newStyle

  ( const String& type )

{
  StyleCtor ctor = CtorMap_::find ( type );

  if ( ctor )
  {
    return ctor ();
  }
  else
  {
    return nullptr;
  }
}


Ref<MarkerStyle> MarkerStyleTable::newStyle

  ( const Properties&  conf,
    const Properties&  props )

{
  using jive::util::Factory;

  Ref<Style>  style;
  String      type;

  props.get ( type, Factory::TYPE_PROP );
  conf .set ( Factory::TYPE_PROP, type );

  style = newStyle ( type );

  if ( ! style )
  {
    Factory::noSuchTypeError (
      props.getContext ( Factory::TYPE_PROP ),
      type,
      "marker style",
      listStyleTypes ()
    );
  }

  return style;
}


//-----------------------------------------------------------------------
//   initOnce_
//-----------------------------------------------------------------------


void MarkerStyleTable::initOnce_ ()
{
  CtorMap_::insert ( "Point",   & newPointMarker   );
  CtorMap_::insert ( "Orb",     & newOrbMarker     );
  CtorMap_::insert ( "Tee",     & newTeeMarker     );
  CtorMap_::insert ( "Number",  & newNumberMarker  );
  CtorMap_::insert ( "Circle",  & newCircleMarker  );
  CtorMap_::insert ( "Diamond", & newDiamondMarker );
  CtorMap_::insert ( "Disc",    & newDiscMarker    );
  CtorMap_::insert ( "Square",  & newSquareMarker  );
}


JIVE_END_PACKAGE( gl )
