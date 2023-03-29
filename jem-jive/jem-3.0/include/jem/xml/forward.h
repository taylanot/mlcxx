
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

#ifndef JEM_XML_FORWARD_H
#define JEM_XML_FORWARD_H

#include <jem/defines.h>


JEM_BEGIN_PACKAGE( xml )


class                       DOMElement;
class                       DOMIterator;
class                       DOMTree;
class                       ParseLog;
class                       ParserException;
class                       Parser;
class                       ParserStack;
class                       ParserState;
class                       Tokenizer;

class                       BeginEmptyTag;
typedef BeginEmptyTag       beginEmptyTag;
struct                      EndEmptyTag;
extern const EndEmptyTag    endEmptyTag;
class                       PrintEmptyTag;
typedef PrintEmptyTag       emptyTag;
class                       BeginStartTag;
typedef BeginStartTag       beginStartTag;
struct                      EndStartTag;
extern const EndStartTag    endStartTag;
class                       PrintStartTag;
typedef PrintStartTag       startTag;
class                       PrintEndTag;
typedef PrintEndTag         endTag;
class                       PrintAttrib;
typedef PrintAttrib         attribute;
struct                      BeginComment;
extern const BeginComment   beginComment;
struct                      EndComment;
extern const EndComment     endComment;
class                       PrintComment;
typedef PrintComment        comment;


JEM_END_PACKAGE( xml )

#endif
