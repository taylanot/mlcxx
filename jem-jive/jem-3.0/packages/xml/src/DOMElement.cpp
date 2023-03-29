
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


#include <jem/base/NullPointerException.h>
#include <jem/xml/DOMAttribs.h>
#include <jem/xml/DOMElement.h>


JEM_BEGIN_PACKAGE( xml )


//=======================================================================
//   class DOMElement
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  DOMElement::TEXT_TYPE_NAME = "$TEXT";
const char*  DOMElement::DATA_TYPE_NAME = "$DATA";


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


DOMElement::DOMElement ()
{}


DOMElement::DOMElement

  ( const String&  typeName,
    const String&  contents ) :

    typeName ( typeName ),
    attribs  ( typeName ),
    parent_  (  nullptr )

{
  attribs.set ( DOMAttribs::TYPE_NAME, typeName );
  attribs.set ( DOMAttribs::CONTENTS,  contents );
}


DOMElement::DOMElement

  ( const String&      typeName,
    const Properties&  attribs ) :

    typeName  ( typeName ),
    attribs   (  attribs ),
    parent_   (  nullptr )

{
  attribs.set ( DOMAttribs::TYPE_NAME, typeName );
}


DOMElement::DOMElement

  ( const String&      typeName,
    const Properties&  attribs,
    const ChildArray&  children ) :

    typeName  ( typeName ),
    attribs   (  attribs ),
    parent_   (  nullptr ),
    children_ ( children )

{
  const idx_t  n = children_.size ();

  for ( idx_t i = 0; i < n; i++ )
  {
    if ( ! children_[i] )
    {
      throw NullPointerException (
        JEM_FUNC,
        "NIL child element specified"
      );
    }

    children_[i]->parent_ = this;
  }

  attribs.set ( DOMAttribs::TYPE_NAME, typeName );
}


DOMElement::~DOMElement ()
{}


//-----------------------------------------------------------------------
//   isEmpty
//-----------------------------------------------------------------------


bool DOMElement::isEmpty () const
{
  if ( children_.size() > 0 )
  {
    return false;
  }
  else
  {
    String contents;

    attribs.find ( contents, DOMAttribs::CONTENTS );

    return (contents.size() == 0);
  }
}


//-----------------------------------------------------------------------
//   getChildren
//-----------------------------------------------------------------------


DOMElement::ChildArray DOMElement::getChildren () const
{
  return children_.clone ();
}


//-----------------------------------------------------------------------
//   getAttribNames
//-----------------------------------------------------------------------


Array<String> DOMElement::getAttribNames () const
{
  Array<String>  names;

  attribs.find ( names, DOMAttribs::ATTRIB_NAMES );

  return names;
}


//-----------------------------------------------------------------------
//   getContents
//-----------------------------------------------------------------------


String DOMElement::getContents () const
{
  String  contents;

  attribs.find ( contents, DOMAttribs::CONTENTS );

  return contents;
}


//-----------------------------------------------------------------------
//   getSource
//-----------------------------------------------------------------------


String DOMElement::getSource () const
{
  String  source;

  attribs.find ( source, DOMAttribs::SOURCE );

  return source;
}


//-----------------------------------------------------------------------
//   getLineNumber
//-----------------------------------------------------------------------


idx_t DOMElement::getLineNumber () const
{
  idx_t  lineno = -1;

  attribs.find ( lineno, DOMAttribs::LINENO );

  return lineno;
}


//-----------------------------------------------------------------------
//   getContext
//-----------------------------------------------------------------------


String DOMElement::getContext () const
{
  String  source;
  idx_t   lineno = -1;

  attribs.find ( source, DOMAttribs::SOURCE );
  attribs.find ( lineno, DOMAttribs::LINENO );

  if ( source.size() > 0 && lineno >= 0 )
  {
    return String::format ( "element `%s\' from `%s\', line %d",
                            typeName,
                            source,
                            lineno );
  }
  else if ( source.size() > 0 )
  {
    return String::format ( "element `%s\' from `%s\'",
                            typeName,
                            source );
  }
  else if ( lineno >= 0 )
  {
    return String::format ( "element `%s\', line %d",
                            typeName,
                            lineno );
  }
  else
  {
    return String::format ( "element `%s\'", typeName );
  }
}


JEM_END_PACKAGE( xml )
