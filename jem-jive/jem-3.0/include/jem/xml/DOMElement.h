
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

#ifndef JEM_XML_DOMELEMENT_H
#define JEM_XML_DOMELEMENT_H

#include <jem/base/Object.h>
#include <jem/util/Properties.h>
#include <jem/xml/import.h>


JEM_BEGIN_PACKAGE( xml )


//-----------------------------------------------------------------------
//   class DOMElement
//-----------------------------------------------------------------------


class DOMElement : public Object
{
 public:

  typedef DOMElement        Self;
  typedef Object            Super;
  typedef Array
    < Ref<Self> >           ChildArray;


  static const char*        TEXT_TYPE_NAME;
  static const char*        DATA_TYPE_NAME;


                            DOMElement ();

                            DOMElement

    ( const String&           typeName,
      const String&           contents );

                            DOMElement

    ( const String&           typeName,
      const Properties&       attribs );

                            DOMElement

    ( const String&           typeName,
      const Properties&       attribs,
      const ChildArray&       children );

  inline bool               isRoot          () const noexcept;
  inline bool               isText          () const noexcept;
  inline bool               isData          () const noexcept;
  bool                      isEmpty         () const;
  inline DOMElement*        getParent       () const noexcept;
  inline idx_t              childCount      () const noexcept;

  inline DOMElement*        getChild

    ( idx_t                   ichild )         const;

  ChildArray                getChildren     () const;
  Array<String>             getAttribNames  () const;
  String                    getContents     () const;
  String                    getSource       () const;
  idx_t                     getLineNumber   () const;
  String                    getContext      () const;


 public:

  String                    typeName;
  Properties                attribs;


 protected:

  virtual                  ~DOMElement      ();


 private:

  DOMElement*               parent_;
  ChildArray                children_;

};






//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   isRoot
//-----------------------------------------------------------------------


inline bool DOMElement::isRoot () const noexcept
{
  return (parent_ == 0);
}


//-----------------------------------------------------------------------
//   isText
//-----------------------------------------------------------------------


inline bool DOMElement::isText () const noexcept
{
  return (typeName == TEXT_TYPE_NAME);
}


//-----------------------------------------------------------------------
//   isData
//-----------------------------------------------------------------------


inline bool DOMElement::isData () const noexcept
{
  return (typeName == DATA_TYPE_NAME);
}


//-----------------------------------------------------------------------
//   getParent
//-----------------------------------------------------------------------


inline DOMElement* DOMElement::getParent () const noexcept
{
  return parent_;
}


//-----------------------------------------------------------------------
//   childCount
//-----------------------------------------------------------------------


inline idx_t DOMElement::childCount () const noexcept
{
  return children_.size ();
}


//-----------------------------------------------------------------------
//   getChild
//-----------------------------------------------------------------------


inline DOMElement* DOMElement::getChild ( idx_t ichild ) const
{
  return children_[ichild].get ();
}


JEM_END_PACKAGE( xml )

#endif
