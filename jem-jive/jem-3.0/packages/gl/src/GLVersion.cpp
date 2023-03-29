
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


#include <cctype>
#include <jem/base/Integer.h>
#include <jem/gl/GLVersion.h>


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class GLVersion
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const GLVersion::Version_ GLVersion::VERSIONS_ [] =
{
  { 0, 0, NO_PROFILE     },
  { 1, 0, COMPAT_PROFILE },
  { 1, 0, CORE_PROFILE   },
  { 1, 1, COMPAT_PROFILE },
  { 1, 1, CORE_PROFILE   },
  { 1, 2, COMPAT_PROFILE },
  { 1, 2, CORE_PROFILE   },
  { 1, 3, COMPAT_PROFILE },
  { 1, 3, CORE_PROFILE   },
  { 1, 4, COMPAT_PROFILE },
  { 1, 4, CORE_PROFILE   },
  { 1, 5, COMPAT_PROFILE },
  { 1, 5, CORE_PROFILE   },
  { 2, 0, COMPAT_PROFILE },
  { 2, 0, CORE_PROFILE   },
  { 2, 1, COMPAT_PROFILE },
  { 2, 1, CORE_PROFILE   },
  { 3, 0, COMPAT_PROFILE },
  { 3, 0, CORE_PROFILE   },
  { 3, 1, COMPAT_PROFILE },
  { 3, 1, CORE_PROFILE   },
  { 3, 2, COMPAT_PROFILE },
  { 3, 2, CORE_PROFILE   },
  { 3, 3, COMPAT_PROFILE },
  { 3, 3, CORE_PROFILE   },
  { 4, 0, COMPAT_PROFILE },
  { 4, 0, CORE_PROFILE   },
  { 4, 1, COMPAT_PROFILE },
  { 4, 1, CORE_PROFILE   },
  { 4, 2, COMPAT_PROFILE },
  { 4, 2, CORE_PROFILE   },
  { 4, 3, COMPAT_PROFILE },
  { 4, 3, CORE_PROFILE   },
  { 4, 4, COMPAT_PROFILE },
  { 4, 4, CORE_PROFILE   },
  { 4, 5, COMPAT_PROFILE },
  { 4, 5, CORE_PROFILE   },
  { 4, 6, COMPAT_PROFILE },
  { 4, 6, CORE_PROFILE   }
};


//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


GLVersion::GLVersion ()
{
  major_     = 0;
  minor_     = 0;
  profile_   = NO_PROFILE;
  versionID_ = VERSION_0;
}


GLVersion::GLVersion ( VersionID vid )
{
  major_     = 0;
  minor_     = 0;
  profile_   = NO_PROFILE;
  versionID_ = vid;

  setVersion ( vid );
}


GLVersion::GLVersion

  ( int      major,
    int      minor,
    Profile  profile )

{
  major_     = major;
  minor_     = minor;
  profile_   = profile;
  versionID_ = getVersionID_ ();
}


GLVersion::GLVersion

  ( const String&  vers,
    Profile        profile )

{
  major_     = 0;
  minor_     = 0;
  profile_   = profile;
  versionID_ = VERSION_0;

  setVersion ( vers );
}


//-----------------------------------------------------------------------
//   setVersion
//-----------------------------------------------------------------------


void GLVersion::setVersion ( VersionID vid )
{
  major_     = (int) VERSIONS_[vid].major;
  minor_     = (int) VERSIONS_[vid].minor;
  profile_   =       VERSIONS_[vid].profile;
  versionID_ = vid;
}


void GLVersion::setVersion

  ( int  major,
    int  minor )

{
  major_     = major;
  minor_     = minor;
  versionID_ = getVersionID_ ();
}


void GLVersion::setVersion ( const String& vers )
{
  const idx_t  len   = vers.size ();
  idx_t        ipos  = 0;
  int          major = 0;
  int          minor = 0;


  while ( ipos < len && std::isdigit( vers[ipos] ) )
  {
    ipos++;
  }

  if ( ipos < len )
  {
    idx_t  jpos = ipos + 1;

    while ( jpos < len && std::isdigit( vers[jpos] ) )
    {
      jpos++;
    }

    Integer::parse ( major, vers[slice(BEGIN,ipos)] );
    Integer::parse ( minor, vers[slice(ipos + 1,jpos) ] );
  }
  else
  {
    Integer::parse ( major, vers );
  }

  major_     = major;
  minor_     = minor;
  versionID_ = getVersionID_ ();
}


//-----------------------------------------------------------------------
//   setProfile
//-----------------------------------------------------------------------


void GLVersion::setProfile ( Profile profile )
{
  profile_   = profile;
  versionID_ = getVersionID_ ();
}


//-----------------------------------------------------------------------
//   toString()
//-----------------------------------------------------------------------


String GLVersion::toString () const
{
  String  vers = String::format ( "%d.%d", major_, minor_ );

  if      ( profile_ == CORE_PROFILE )
  {
    vers = vers + " [Core Profile]";
  }
  else if ( profile_ == COMPAT_PROFILE )
  {
    vers = vers + " [Compatibility Profile]";
  }

  return vers;
}


//-----------------------------------------------------------------------
//   isValid
//-----------------------------------------------------------------------


bool GLVersion::isValid () const
{
  if ( minor_ < 0 )
  {
    return false;
  }

  switch ( major_ )
  {
  case 1:

    return (minor_ <= 5);

  case 2:

    return (minor_ <= 1);

  case 3:

    return (minor_ <= 3);

  case 4:

    return (minor_ <= 6);
  }

  return false;
}


//-----------------------------------------------------------------------
//   getVersionID_
//-----------------------------------------------------------------------


GLVersion::VersionID GLVersion::getVersionID_ () const
{
  for ( int vid = 0; vid <= LAST_VERSION; vid++ )
  {
    if ( (major_   == (int) VERSIONS_[vid].major) &&
         (minor_   == (int) VERSIONS_[vid].minor) &&
         (profile_ ==       VERSIONS_[vid].profile) )
    {
      return (VersionID) vid;
    }
  }

  return VERSION_0;
}


JEM_END_PACKAGE( gl )

