
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


#include <jem/base/rtti.h>
#include <jem/base/Error.h>
#include <jem/util/ObjFlex.h>
#include <jem/util/DictionaryEnumerator.h>
#include <jem/xml/print.h>
#include <jem/xml/DOMAttribs.h>
#include <jem/xml/DOMIterator.h>
#include <jem/xml/DOMTree.h>


JEM_BEGIN_PACKAGE( xml )


using jem::util::ObjFlex;
using jem::util::DictEnum;


//=======================================================================
//   class DOMTree
//=======================================================================

//-----------------------------------------------------------------------
//   print
//-----------------------------------------------------------------------


void DOMTree::print

  ( PrintWriter&       out,
    const DOMElement&  root,
    PrintOptions       options )

{
  using jem::xml::print;

  DOMIterator  iter ( const_cast<DOMElement*>( &root ) );
  DOMElement*  elem;

  for ( elem = iter.get(); elem; elem = iter.next() )
  {
    if ( iter.dir() < 0 )
    {
      printElement_ ( out, *elem, options );
    }
    else
    {
      print ( out, endTag( elem->typeName ) );
    }
  }
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<DOMElement> DOMTree::clone

  ( const DOMElement&   root,
    CloneOptions        options )

{
  DOMIterator      iter ( const_cast<DOMElement*>( &root ) );

  Ref<DOMElement>  copy;
  ObjFlex          stack;
  DOMElement*      elem;


  stack.reserve ( 16 );

  for ( elem = iter.get(); elem; elem = iter.next() )
  {
    if ( iter.dir() < 0 && elem->childCount() == 0 )
    {
      if ( options & CLONE_DEEP )
      {
        copy = newInstance<DOMElement> (
          elem->typeName,
          elem->attribs.clone ()
        );
      }
      else
      {
        copy = newInstance<DOMElement> (
          elem->typeName,
          elem->attribs
        );
      }

      stack.pushBack ( copy );
    }
    else if ( iter.dir() > 0 )
    {
      Array< Ref<DOMElement> >  children ( elem->childCount() );

      idx_t  n = elem->childCount ();
      idx_t  j = stack.size() - n;

      for ( idx_t i = 0; i < n; i++, j++ )
      {
        children[i] = stack.getAs<DOMElement> ( j );
      }

      stack.popBack ( n );

      if ( options & CLONE_DEEP )
      {
        copy = newInstance<DOMElement> (
          elem->typeName,
          elem->attribs.clone (),
          children
        );
      }
      else
      {
        copy = newInstance<DOMElement> (
          elem->typeName,
          elem->attribs,
          children
        );
      }

      stack.pushBack ( copy );
    }
  }

  if ( stack.size() != 1 )
  {
    throw Error (
      JEM_FUNC,
      "invalid stack size: " + String ( stack.size() )
    );
  }

  return stack.getAs<DOMElement> ( 0 );
}


//-----------------------------------------------------------------------
//   printAttribs_
//-----------------------------------------------------------------------


void DOMTree::printAttribs_

  ( PrintWriter&        out,
    const Properties&   attribs,
    int                 options )

{
  using jem::print;
  using jem::io::space;
  using jem::io::print;
  using jem::xml::print;

  Ref<DictEnum>  e = attribs.enumerate ();

  for ( ; ! e->atEnd(); e->toNext() )
  {
    String  name = e->getKey ();

    if ( name.size() == 0 || name[0] == DOMAttribs::PREFIX )
    {
      continue;
    }

    Ref<Object>  obj = e->getValue ();

    print ( out, space, name, "=\"" );

    if      ( ! obj )
    {
      print ( out, "[]" );
    }
    else if ( (options & QUOTE_STRINGS) &&
              isInstance<String>( obj ) )
    {
      String  str = toValue<String> ( obj );

      print ( out, '\'', str, '\'' );
    }
    else
    {
      if ( ! Properties::printValue( out, "", obj ) )
      {
        print ( out, obj.toString() );
      }
    }

    print ( out, '\"' );
  }
}


//-----------------------------------------------------------------------
//   printDOMElement
//-----------------------------------------------------------------------


void DOMTree::printElement_

  ( PrintWriter&       out,
    const DOMElement&  elem,
    int                options )

{
  using jem::print;
  using jem::io::space;
  using jem::io::print;
  using jem::io::blankLine;
  using jem::xml::print;

  String  contents;

  elem.attribs.find ( contents, DOMAttribs::CONTENTS );

  if ( elem.childCount() == 0 && contents.size() == 0 )
  {
    print ( out, blankLine );
    print ( out, beginEmptyTag( elem.typeName ) );

    printAttribs_ ( out, elem.attribs, options );

    print ( out, endEmptyTag );
    print ( out, blankLine   );
  }
  else if ( elem.isText() && contents.size() > 0 )
  {
    print ( out, blankLine );
    print ( out, contents  );
    print ( out, blankLine );
  }
  else
  {
    print ( out, beginStartTag( elem.typeName ) );

    printAttribs_ ( out, elem.attribs, options );

    print ( out, endStartTag );

    if ( contents.size() > 0 )
    {
      print ( out, blankLine );
      print ( out, contents  );
      print ( out, blankLine );
    }

    if ( elem.childCount() == 0 )
    {
      print ( out, endTag( elem.typeName ) );
    }
  }
}


JEM_END_PACKAGE( xml )
