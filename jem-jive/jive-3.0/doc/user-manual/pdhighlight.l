/*
 *  This is a flex input file but should be edited in -*-C-*- mode
 *
 *  C++2LaTeX: Produce prettyprinted LaTeX files from  C++ or C sources.
 *  Copyright (C) 1990 Norbert Kiesel
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 1, or (at your option)
 *  any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 *
 *  Norbert Kiesel     
 *  RWTH Aachen / Institut f. Informatik III
 *  Ahornstr. 55
 *  D-5100 Aachen
 *  West Germany  
 *
 *  Phone:  +49 241 80-7266
 *  EUNET:  norbert@rwthi3.uucp
 *  USENET: ...!mcvax!unido!rwthi3!norbert
 *  X.400:  norbert@rwthi3.informatik.rwth-aachen.de
 *
 *  Please contact me for any bugs you find in this code or any
 *  improvements! I'd also be very happy to get feedback where and
 *  how frequently this program is used (just drop a little mail :-).
 */

/* we use the builtin start condition stack of flex */

%option stack

/* exclusive start conditions */

%x STRING INCLUDE REF SPECIAL SPECIAL2 SPECIAL3 XMLDECLS

/* inclusive start conditions */

%s BCOMMENT XMLCOMMENT SCMCOMMENT ELCOMMENT C CPP JAVA PLAIN XML SCM EL

%{
enum Mode {C_MODE=0, CPP_MODE, JAVA_MODE, AWK_MODE, TEXT_MODE, 
	   XML_MODE, SCM_MODE, EL_MODE };
extern int mode;

/* try to get rid of all this defines - replace them by functions (vhs)

#define KEY	  printf ("{%s %s}", keyword_font, yytext)
#define CPrePr	  printf ("{%s \\%s}", cpp_font, yytext)
#define OUT(x)	  printf ("%s", x)
#define SUB(x)	  substitute(x)

*/

#define IND	  indent(yytext)
#define BSLASH    printf("\\symbol{92}")
#define LAT_beg   printf("\\begin{latexonly}%%\n")
#define LAT_end   printf("%%\n\\end{latexonly}%%\n")
#define HTML_beg  printf("\\begin{htmlonly}%%\n")
#define HTML_end  printf("%%\n\\end{htmlonly}%%\n")

void INIT(void);                          /* declared before main() */

#include <stdio.h>

int mode = CPP_MODE;                      /* C++ is the default mode */

int center                = 0;
int frame                 = 0;
int nohtml                = 1;
int complete_file         = 0;
int keyword_underline     = 0;
int header                = 0;
int tabtotab              = 8;
int piped                 = 0;
int want_index            = 0;
char * font_size          = "\\pdFontSize";     /* "10";       */
char * line_sep           = "\\pdLineSep";      /* "2.5";      */
                                                /* vhs         */
char * indentation        = "\\pdIndent";       /* "0.5em";    */
char * base_font          = "\\pdBaseFont";     /* "cmr";      */
char * comment_font       = "\\pdCommentFont";  /* "itshape";  */
char * keyword_font       = "\\pdKeywordFont";  /* "bfseries"; */
char * header_font        = "\\pdHeaderFont";   /* "slshape";  */
char * cpp_font           = "\\pdPreprFont";    /* "ttfamily"; */
char * string_font        = "\\pdStringFont";   /* "ttfamily"; */
char * listing_name       = NULL;
char * label_name         = NULL;
char * default_label_name = NULL;
char * reference_name     = NULL;
char * line_begin         = NULL;
char * line_end           = NULL;
char * new_name;
char * temp;
char   indent_buf[512];

/* html support */
/* <SPAN STYLE="font-weight : bold; font-style : italic; color : purple; "> */

char * html_KEY_begin  = "<SPAN STYLE=\"color : maroon; \">";
char * html_KEY_end    = "</SPAN>";
char * html_COM_begin  = "<SPAN STYLE=\"color : red; \">";
char * html_COM_end    = "</SPAN>";
char * html_STR_begin  = "<SPAN STYLE=\"color : teal; \">";
char * html_STR_end    = "</SPAN>";
char * html_CPP_begin  = "<SPAN STYLE=\"color : green; \">";
char * html_CPP_end    = "</SPAN>";
char * html_color      = "black";
char * html_background = "white";
FILE * html_file;

char* substitute     (const char *);
char* html_substitute(const char *);
void  indent         (const char *);
void  indent_string  (char *, const char *);
void  newpage        (int);
void  usage          (const char *);

void KEY() {

  if (keyword_underline)
    printf ("{%s \\underline{%s}}", keyword_font, substitute(yytext));
  else
    printf ("{%s %s}", keyword_font, substitute(yytext));

  fprintf(html_file, "%s%s%s", html_KEY_begin, 
	  html_substitute(yytext), html_KEY_end);

}

%}

%%

      INIT();  /* ATTENTION : must not begin in the first column */

 /* Any content enclosed between '/*?' and '?*\/' is written without         */
 /* any special handling. NOTE : '/' must be escaped in the closing comment! */

[ \t]+"/*?"                   { /* We save the leading whitespaces in */
                                /* 'indent_buf'                       */
                                indent_string(indent_buf, yytext);
                                yy_push_state(SPECIAL); 
                              }
<SPECIAL>"?*/"                { yy_pop_state(); 
                              } 
<SPECIAL>"\\begin{latexonly}" { printf("%s\n\\hspace*{%d\\indentation}", 
                                  yytext, strlen(indent_buf)); 
                                yy_push_state(SPECIAL2);
                              }
<SPECIAL2>"\\end{latexonly}"  { printf("%s", yytext); yy_pop_state(); 
                              }
<SPECIAL2>[\n]                |
<SPECIAL2>.                   { printf("%s", yytext); }
<SPECIAL>"\\begin{htmlonly}"  { fprintf(html_file, "\n\\end{rawhtml}\n"); 
                
                fprintf(html_file, "%s\n", yytext);
				fprintf(html_file,"\\rawhtml{%s}", indent_buf);
			        yy_push_state(SPECIAL3); 
                              }
<SPECIAL3>"\\end{htmlonly}"   { yy_pop_state();
                                fprintf(html_file, "%s", yytext);
			        fprintf(html_file, "\n\\begin{rawhtml}"); 
                              }
<SPECIAL3>[\n]                |
<SPECIAL3>.                   { fprintf(html_file, "%s", yytext); 
                              }

<PLAIN>[ \t]+		       IND; 
<PLAIN>"\f"[\n]?	      { printf("\\newpage\n");
			        fprintf(html_file, "\n"); 
                              }
<PLAIN>"\n"		      { printf("\\mbox{}\\\\\n");
				fprintf(html_file, "\n"); 
                              }
<PLAIN>.                      { printf("%s", substitute(yytext));
				fprintf(html_file, "%s", yytext); 
                              }


<C,CPP>"#"[ \t]*"include"     { printf ("{%s \\%s}", cpp_font, yytext);
                                fprintf(html_file, "%s%s", 
                                        html_CPP_begin, yytext);
			         yy_push_state(INCLUDE); 
                              }
<C,CPP>"#"[ \t]*"define"      |
<C,CPP>"#"[ \t]*"undef"	      |
<C,CPP>"#"[ \t]*"pragma"      |
<C,CPP>"#"[ \t]*"if"	      |
<C,CPP>"#"[ \t]*"ifdef"	      |
<C,CPP>"#"[ \t]*"ifndef"      |
<C,CPP>"#"[ \t]*"elif"	      |
<C,CPP>"#"[ \t]*"else"	      |
<C,CPP>"#"[ \t]*"error"	      |
<C,CPP>"#"[ \t]*"endif"	      |
<C,CPP>"#"[ \t]*"line"	      { printf ("{%s \\%s}", cpp_font, yytext);
                                fprintf(html_file, "%s%s%s", html_CPP_begin, 
                                        yytext, html_CPP_end); 
                              }


<INCLUDE>"<"[^>]*/">"	      { printf("<{%s{}%s}>", 
			        string_font, substitute(yytext+1));
                                fprintf(html_file, "<%s%s%s>%s",
				        html_STR_begin, substitute(yytext+1),
				        html_STR_end, html_CPP_end);
			        if (want_index)
			          printf("\\index{%s}", substitute(yytext+1));
			        input(); 
                                yy_pop_state(); 
                              }
<INCLUDE>\"[^\"]*/\"	      { printf("\\symbol{34}{%s{}%s}\\symbol{34}", 
				  string_font, substitute(yytext+1));
                                fprintf(html_file, "\"%s%s%s\"%s",
				        html_STR_begin, substitute(yytext+1),
				        html_STR_end, html_CPP_end);
			        if (want_index)
			          printf("\\index{%s}", substitute(yytext+1));
			        input(); 
                                yy_pop_state(); 
                              }
<INCLUDE>[ \t]+		      { printf("%s", yytext);
                                fprintf(html_file, "%s", yytext); 
                              }
<INCLUDE>[\n]		      printf("\\mbox{}\\\\\n");
<INCLUDE>.		      { yyless (0); 
                                yy_pop_state(); 
                              }

<C,CPP>"auto"		|
<C,CPP,JAVA>"double"	|
<C,CPP,JAVA>"int"	|
<C,CPP>"struct"		|
<C,CPP,JAVA>"break"	|
<C,CPP,JAVA>"else"	|
<C,CPP,JAVA>"long"	|
<C,CPP,JAVA>"switch"	|
<C,CPP,JAVA>"case"	|
<C,CPP>"enum"		|
<C,CPP>"register"	|
<C,CPP>"typedef"	|
<C,CPP,JAVA>"char"	|
<C,CPP>"extern"		|
<C,CPP,JAVA>"return"	|
<C,CPP>"union"		|
<C,CPP,JAVA>"const"	|
<C,CPP,JAVA>"float"	|
<C,CPP,JAVA>"short"	|
<C,CPP>"unsigned"	|
<C,CPP,JAVA>"continue"	|
<C,CPP,JAVA>"for"	|
<C,CPP>"signed"		|
<C,CPP,JAVA>"void"	|
<C,CPP,JAVA>"default"	|
<C,CPP,JAVA>"goto"	|
<C,CPP>"sizeof"		|
<C,CPP,JAVA>"volatile"	|
<C,CPP,JAVA>"do"	|
<C,CPP,JAVA>"if"	|
<C,CPP,JAVA>"static"	|
<C,CPP,JAVA>"while"	|
<CPP>"explicit"		|
<CPP>"bool"		|
<CPP,JAVA>"new"		|
<CPP>"delete"		|
<CPP,JAVA>"this"	|
<CPP,JAVA>"operator"	|
<CPP,JAVA>"class"	|
<CPP,JAVA>"public"	|
<CPP,JAVA>"protected"	|
<CPP,JAVA>"private"	|
<CPP>"virtual"		|
<CPP>"friend"		|
<CPP>"inline"		|
<CPP>"namespace"	|
<CPP>"using"		|
<CPP>"template"		|
<CPP>"typename"		|
<CPP>"typeid"		|
<CPP>"const_cast"       |
<CPP>"static_cast"      |
<CPP>"dynamic_cast"     |
<CPP>"reinterpret_cast" |
<CPP,JAVA>"throw"	|
<CPP,JAVA>"throws"	|
<CPP,JAVA>"try"		|
<CPP,JAVA>"catch"	|
<JAVA>"finally"		|
<JAVA>"import"		|
<JAVA>"boolean"		|
<JAVA>"false"		|
<JAVA>"true"		|
<JAVA>"byte"		|
<JAVA>"byvalue"		|
<JAVA>"cast"		|
<JAVA>"future"		|
<JAVA>"generic"		|
<JAVA>"inner"		|
<JAVA>"null"		|
<JAVA>"outer"		|
<JAVA>"rest"		|
<JAVA>"super"		|
<JAVA>"var"		|
<JAVA>"package"		|
<JAVA>"extends"		|
<JAVA>"abstract"	|
<JAVA>"interface"	|
<JAVA>"implements"	|
<JAVA>"final"		|
<JAVA>"native"		|
<JAVA>"synchronized"	|
<JAVA>"instanceof"	|
<JAVA>"transient"	|
<JAVA>"overload"	KEY();


<CPP>"->*"		|
<CPP>".*"		|
<JAVA>">>>="		|
<JAVA>">>>"		|
<C,CPP>"->"		|
<C,CPP>"..."		|
<C,CPP,JAVA>"<<"	|
<C,CPP,JAVA>">>"	|
<C,CPP,JAVA>"<="	|
<C,CPP,JAVA>">="	|
<C,CPP,JAVA>"!="	|
<C,CPP,JAVA>"||"	|
<C,CPP,JAVA>"*="	|
<C,CPP,JAVA>"<<="	|
<C,CPP,JAVA>">>="	|
<C,CPP,JAVA>"|="	|
<C,CPP,JAVA>"*"		|
<C,CPP,JAVA>"|"		|
<C,CPP,JAVA>"/"		|
<C,CPP,JAVA>"<"		|
<C,CPP,JAVA>">"		|
<C,CPP,JAVA>"?"		|
<C,CPP,JAVA>":"		|
<C,CPP,JAVA>"="		|
<C,CPP,JAVA>","		|
<C,CPP,JAVA>"."		|
<C,CPP,JAVA>";"		|
<C,CPP,JAVA>"!"		|
<C,CPP,JAVA>"-"		|
<C,CPP,JAVA>"+"		|
<C,CPP,JAVA>"/="	|
<C,CPP,JAVA>"=="	|
<C,CPP,JAVA>"++"	|
<C,CPP,JAVA>"+="	|
<C,CPP,JAVA>"-="	|
<C,CPP,JAVA>"("		|
<C,CPP,JAVA>")"		|
<C,CPP,JAVA>"["		|
<C,CPP,JAVA>"]"		|
<CPP>"::"		{ printf("%s", yytext); 
                          fprintf(html_file, "%s", html_substitute(yytext)); 
                        }

<C,CPP,JAVA>"^"		|
<C,CPP,JAVA>"^="	|
<C,CPP,JAVA>"~"		|
<C,CPP,JAVA>"&&"	|
<C,CPP,JAVA>"%="	|
<C,CPP,JAVA>"&="	|
<C,CPP,JAVA>"{"		|
<C,CPP,JAVA>"}"		|
<C,CPP,JAVA>"&"		|
<C,CPP,JAVA>"%"		{ printf("%s", substitute(yytext));
                          fprintf(html_file, "%s", html_substitute(yytext)); 
                        }
<C,CPP,JAVA>"--"	{ printf("-{}-");
                          fprintf(html_file, "%s", yytext); 
                        }


<C,CPP,JAVA>[a-zA-Z_][a-zA-Z_0-9]*    { printf("%s", substitute(yytext));
                                        fprintf(html_file, "%s", yytext);
                                        if (want_index)
					  printf("\\index{%s}", 
						 substitute(yytext)); }

<C,CPP,JAVA>"/*"        { yy_push_state(BCOMMENT);
			  printf("{/*%s{}", comment_font); 
			  fprintf(html_file, "%s/*", html_COM_begin); }
<BCOMMENT>"*/"		{ yy_pop_state(); 
                          printf("*/}"); 
			  fprintf(html_file, "*/%s", html_COM_end); }
<BCOMMENT>"\n"	        { printf("}\\mbox{}\\\\\n{%s{}", comment_font);
			  fprintf(html_file, "\n"); }
<BCOMMENT>[ \t]+	IND;
<BCOMMENT>.		{ printf("%s", substitute(yytext));
                          fprintf(html_file, "%s", yytext); }

<CPP,JAVA>"//".*$	{ printf("{//%s{}%s}",
				 comment_font, substitute(yytext + 2));
			  fprintf(html_file, "%s%s%s",
				  html_COM_begin, yytext, html_COM_end); }

<CPP>L?\" 		{ yy_push_state(STRING);
			  printf("{%s\\symbol{34}",string_font); 
			  fprintf(html_file, "%s\"", html_STR_begin); }
<C,JAVA,XML,SCM,EL>\"	{ yy_push_state(STRING);
			  printf("{%s\\symbol{34}",string_font); 
			  fprintf(html_file, "%s\"", html_STR_begin); }
<STRING>\\\\		{ BSLASH; BSLASH; 
                          fprintf(html_file, "\\\\"); }
<STRING>\\[bfnrtv'"]	{ BSLASH;  printf("%s", substitute(yytext+1)); 
                          fprintf(html_file, "\\%s", yytext+1); }
<STRING>\"		{ yy_pop_state(); printf("\\symbol{34}}"); 
			  fprintf(html_file, "\"%s", html_STR_end); }
<STRING>"\n"		{ printf("}\\mbox{}\\\\\n{%s{}", string_font);
			  fprintf(html_file, "\n"); }
<STRING>^[ \t]+		IND;
<STRING>.		{ printf("%s", substitute(yytext));
                          fprintf(html_file, "%s", yytext); }

<C,CPP,JAVA>([0-9]*\.[0-9]+[fFlL]?)		|
<C,CPP,JAVA>([0-9]+\.[0-9]*[fFlL]?)		|
<C,CPP,JAVA>([0-9]*\.?[0-9]+[eE][+-]?[0-9]+)	|
<C,CPP,JAVA>([0-9]+\.?[0-9]*[eE][+-]?[0-9]+)    { printf("%s", yytext);
                                                  fprintf(html_file, "%s", 
                                                                     yytext); }

<C,CPP,JAVA>[0-9]+[uUlL]?		{ printf("%s", yytext);
                                          fprintf(html_file, "%s", yytext); }

<C,CPP,JAVA>L?'[ -~]'			|
<C,CPP,JAVA>L?'\\[ntvbrfa\\?'"]'	|
<C,CPP,JAVA>L?'\\[0-7]{1,3}'		|
<C,CPP,JAVA>L?'\\x[0-9a-fA-F]{1,2}'	{ printf("%s", substitute(yytext));
                                          fprintf(html_file, "%s", yytext); }

<C,CPP,JAVA>0x[0-9a-fA-F]+[uUlL]?	{ printf("%s", yytext);
                                          fprintf(html_file, "%s", yytext); }

<C,CPP,JAVA>"\\\n"			{ BSLASH; printf("\\\\\n"); 
                                          fprintf(html_file, "\\\n"); }
<C,CPP,JAVA>^[ \t]+			IND;
<C,CPP,JAVA>[ \t]+			IND;  /* vhs : replaced ECHO with */
      /* IND in order to indent comments correctly at the end of the line */
<C,CPP,JAVA>"\f"[\n]?			{ printf("\\newpage\n");
				          fprintf(html_file, "\n"); }
<C,CPP,JAVA>"\n"			{ printf("\\mbox{}\\\\\n");
				          fprintf(html_file, "\n"); }

<XML>"<!--"         { yy_push_state(XMLCOMMENT); 
                      printf("{\\pdCommentFont"); 
		      KEY();
		      fprintf(html_file, "%s", html_COM_begin); }
<XMLCOMMENT>"-->"   { yy_pop_state(); 
                      fprintf(html_file, "%s", html_COM_end); 
		      KEY(); 
		      printf("}"); }
<XMLCOMMENT>"\n"    { printf("\\mbox{}\\\\\n");
                      fprintf(html_file, "\n"); }
<XMLCOMMENT>[ \t]+  IND;
<XMLCOMMENT>.       { printf("%s", substitute(yytext));
                      fprintf(html_file, "%s", html_substitute(yytext)); }


<XML>"<?"[a-zA-Z]*[-a-zA-Z_:.]*[a-zA-Z_]*  |
<XML>"</"[a-zA-Z]*[-a-zA-Z_:.]*[a-zA-Z_]*  |
<XML>"<"[a-zA-Z]*[-a-zA-Z_:.]*[a-zA-Z_]*   KEY();

<XML>"<!"("ENTITY"|"DOCTYPE"|"ELEMENT"|"ATTLIST") { 
     yy_push_state(XMLDECLS);
     printf("{\\pdKeywordFont ");
     if (keyword_underline) printf("\\underline{%s}", substitute(yytext));
     else printf("%s", substitute(yytext)); 
     fprintf(html_file, "%s%s ", html_KEY_begin, html_substitute(yytext)); }
<XMLDECLS>[a-zA-Z]*[-a-zA-Z_:.]*[a-zA-Z_]*        { 
     yy_pop_state();
     printf("{\\pdCommentFont %s}}", substitute(yytext));
     fprintf(html_file, "%s%s", yytext, html_KEY_end); }

<XML>("/>"|">"|"?>") { KEY(); }
<XML>&[^;]*;         { printf("{\\pdCommentFont %s}", substitute(yytext));
                       fprintf(html_file, "%s%s%s", html_COM_begin, 
			       html_substitute(yytext), html_COM_end); }


<XML>"\n"      { printf("\\mbox{}\\\\\n");
                 fprintf(html_file, "\n"); }
<XML>[ \t]+    IND;
<XML>.         { printf("%s", substitute(yytext));
                 fprintf(html_file, "%s", html_substitute(yytext)); }

<SCM>";"+          { yy_push_state(SCMCOMMENT); 
                      printf("{\\pdCommentFont"); 
		      KEY();
		      fprintf(html_file, "%s", html_COM_begin); }
<SCMCOMMENT>"\n"   { yy_pop_state(); 
                      fprintf(html_file, "%s", html_COM_end); 
		      KEY(); 
		      printf("}\\mbox{}\\\\\n"); }
<SCMCOMMENT>[ \t]+  IND;
<SCMCOMMENT>.       { printf("%s", substitute(yytext));
                      fprintf(html_file, "%s", html_substitute(yytext)); }
<SCM>{
  "else" |
  "define" |
  "unquote" |
  "unquote-splicing" |
  "quote" |
  "lambda" |
  "if" |
  "set!" |
  "begin" |
  "cond" |
  "and" |
  "or" |
  "case" |
  "let" |
  "let*" |
  "letrec" |
  "do" |
  "delay" |
  "quasiquote" |
  "#t" |
  "#f"    KEY();

  "\n"      { printf("\\mbox{}\\\\\n");
              fprintf(html_file, "\n"); }
  [ \t]+    IND;

  [a-zA-Z_][a-zA-Z_0-9]* { 
    printf("%s", substitute(yytext));
    fprintf(html_file, "%s", yytext); }

  .         { printf("%s", substitute(yytext));
              fprintf(html_file, "%s", html_substitute(yytext)); }
}

<EL>";"+          { yy_push_state(ELCOMMENT); 
                      printf("{\\pdCommentFont"); 
		      KEY();
		      fprintf(html_file, "%s", html_COM_begin); }
<ELCOMMENT>"\n"   { yy_pop_state(); 
                      fprintf(html_file, "%s", html_COM_end); 
		      KEY(); 
		      printf("}\\mbox{}\\\\\n"); }
<ELCOMMENT>[ \t]+  IND;
<ELCOMMENT>.       { printf("%s", substitute(yytext));
                      fprintf(html_file, "%s", html_substitute(yytext)); }
<EL>{
  "and" |
  "catch" |
  "cond" |
  "condition-case" |
  "defconst" |
  "defmacro" |
  "defun" |
  "defvar" |
  "function" |
  "if" |
  "interactive" |
  "let" |
  "let*" |
  "or" |
  "prog1" |
  "prog2" |
  "progn" |
  "quote" |
  "save-excursion" |
  "save-restriction" |
  "save-window-excursion" |
  "setq" |
  "setq-default" |
  "track-mouse" |
  "unwind-protect" |
  "while" |
  "with-output-to-temp-buffer"    KEY();

  "\n"      { printf("\\mbox{}\\\\\n");
              fprintf(html_file, "\n"); }
  [ \t]+    IND;

  [a-zA-Z_][a-zA-Z_0-9]* { 
    printf("%s", substitute(yytext));
    fprintf(html_file, "%s", yytext); }

  .         { printf("%s", substitute(yytext));
              fprintf(html_file, "%s", html_substitute(yytext)); }
}
%%

/* STRANGE additional OPTION for IDENTIFIERS (perhaps i will need it ??? (vhs)
<C,CPP,JAVA>[a-zA-Z_$][a-zA-Z_$0-9]*
*/

char* substitute (const char * input)
{
  static char buf[1025];
  int x = 0;  
  while (*input && x < 1013) {
    switch (*input) {
    case '_':
    case '&':
    case '#':
    case '$':
    case '%':
    case '{':
    case '}':
      buf[x++] = '\\';
      buf[x++] = *input;
      break;
    case '\\':
      buf[x++] = '\\';
      buf[x++] = 's';
      buf[x++] = 'y';
      buf[x++] = 'm';
      buf[x++] = 'b';
      buf[x++] = 'o';
      buf[x++] = 'l';
      buf[x++] = '{';
      buf[x++] = '9';
      buf[x++] = '2';
      buf[x++] = '}';
      break;
    case '^':
      buf[x++] = '\\';
      buf[x++] = 's';
      buf[x++] = 'y';
      buf[x++] = 'm';
      buf[x++] = 'b';
      buf[x++] = 'o';
      buf[x++] = 'l';
      buf[x++] = '{';
      buf[x++] = '9';
      buf[x++] = '4';
      buf[x++] = '}';
      break;
    case '~':
      buf[x++] = '\\';
      buf[x++] = 's';
      buf[x++] = 'y';
      buf[x++] = 'm';
      buf[x++] = 'b';
      buf[x++] = 'o';
      buf[x++] = 'l';
      buf[x++] = '{';
      buf[x++] = '1';
      buf[x++] = '2';
      buf[x++] = '6';
      buf[x++] = '}';
      break;
    default:
      buf[x++] = *input;
      break;
    }
    input++;
  }
  buf[x] = '\0';
  return buf;
}

char* html_substitute (const char * input)
{
  static char buf[1025];
  int x = 0;  
  while (*input && x < 1013) {
    switch (*input) {
      /* latex2html seem's to have some problems with this ...
    case '"':
      buf[x++] = '&';
      buf[x++] = '#';
      buf[x++] = '3';
      buf[x++] = '4';
      buf[x++] = ';';
      break;
      */
    case '&':
      buf[x++] = '&';
      buf[x++] = '#';
      buf[x++] = '3';
      buf[x++] = '8';
      buf[x++] = ';';
      break;
    case '<':
      buf[x++] = '&';
      buf[x++] = '#';
      buf[x++] = '6';
      buf[x++] = '0';
      buf[x++] = ';';
      break;
    case '>':
      buf[x++] = '&';
      buf[x++] = '#';
      buf[x++] = '6';
      buf[x++] = '2';
      buf[x++] = ';';
      break;
    default:
      buf[x++] = *input;
      break;
    }
    input++;
  }
  buf[x] = '\0';
  return buf;
}

void
indent(const char * blanks)
{
  int i;
  
  i = 0;
  while (*blanks)
    {
      if (*blanks == ' ')
	{
	  i++;
	}
      else			/* *blanks == '\t' */
	{
	  while (++i % tabtotab) ;
	}
      blanks++;
    }
  printf ("\\hspace*{%d\\indentation}", i);
  while (i--) fprintf(html_file, " ");
}

/* Copy leading whitespaces of 'src' to 'dest'. Expand tabs to spaces. */ 
void indent_string(char *dest, const char *src) {
  int i = 0;
  while (*src) {
    if (*src == ' ') i++;
    else if (*src == '\t') while (++i % tabtotab) ;
    else break;
    src++;
  }
  while (i--) *(dest++) = ' ';
  *dest = '\0';
}

#include "getopt.h"
#include <string.h>
#include <fcntl.h>
#include <ctype.h>
#include <time.h>

const char *version_string =
   "pdhighlight 1.0 (based on c++2latex 1.1 Patchlevel 5)";

static struct option opts[] =
{
  {"ansi-c",            0, 0, 'a'},
  {"c-mode",            0, 0, '1'},
  {"c-plusplus",        0, 0, 'p'},
  {"cpp-mode",          0, 0, '2'},
  {"java-mode",         0, 0, '3'},
  {"text-mode",         0, 0, '4'},
  {"xml-mode",          0, 0, '5'},
  {"scheme-mode",       0, 0, '6'},
  {"elisp-mode",        0, 0, '7'},
  {"complete-file",     0, 0, 'c'},
  {"font-size",         1, 0, 's'},
  {"line-sep",          1, 0, 'l'},
  {"line-begin",        1, 0, 'b'},
  {"line-end",          1, 0, 'e'},
  {"indentation",       1, 0, 'i'},
  {"header",            0, 0, 'h'},
  {"center",            0, 0, 'z'},
  {"frame",             0, 0, 'f'},
  {"piped",             0, 0, 't'},
  {"output",            1, 0, 'o'},
  {"keyword_underline", 0, 0, 'u'},
  {"tabstop",           1, 0, 'T'},
  {"base-font",         1, 0, 'F'},
  {"listing-name",      1, 0, 'N'},
  {"label-name",        1, 0, 'L'},
  {"reference-name",    1, 0, 'R'},
  {"comment-font",      1, 0, 'C'},
  {"string-font",       1, 0, 'S'},
  {"keyword-font",      1, 0, 'K'},
  {"header-font",       1, 0, 'H'},
  {"cpp-font",          1, 0, 'P'},
  {"version",           0, 0, 'V'},
  {"index",             0, 0, 'x'},
  {0,                   0, 0, 0}
};


void INIT()
{
  switch (mode) {
  case C_MODE    : yy_push_state(C);     break;
  case CPP_MODE  : yy_push_state(CPP);   break;
  case JAVA_MODE : yy_push_state(JAVA);  break;
  case TEXT_MODE : yy_push_state(PLAIN); break;
  case XML_MODE  : yy_push_state(XML);   break;
  case SCM_MODE  : yy_push_state(SCM);   break;
  case EL_MODE   : yy_push_state(SCM);   break;
  default        : yy_push_state(PLAIN);
  }
}


int main (int argc, char** argv)
{
  int    c;
  int    index;
  int    has_filename;
  char * input_name;
  char * output_name;
  char * program_name;
  time_t now;
  char * today;
  char   buf[513];

  /* open file for temporary html output */

  html_file = fopen("_pd_html.html", "w+");

  input_name   = "Standard Input";
  output_name  = 0;
  now          = time(0);
  today        = ctime(&now);
  program_name = strrchr (argv[0], '/');

  if (program_name == NULL)	/* no pathname */
    {
      program_name = argv[0];
    }
  else
    {
      program_name++;
    }
  
  while ((c = getopt_long (argc, argv,
			   "b:e:a1234567cfuzpo:s:l:i:thT:C:F:N:L:D:R:H:S:K:P:Vx",opts, &index))
	 != EOF)
    {
      if (c == 0)		/* Long option */
	{
	  c = opts[index].val;
	}
      switch (c)
	{
	case '1':
	  mode = C_MODE;
	  break;
	case '2':
	  mode = CPP_MODE;
	  break;
	case '3':
	  mode = JAVA_MODE;
	  break;
	case '4':
	  mode = TEXT_MODE;
	  break;
	case '5':
	  mode = XML_MODE;
	  break;
	case '6':
	  mode = SCM_MODE;
	  break;
	case '7':
	  mode = EL_MODE;
	  break;
	case 'a':
	  mode = C_MODE;
	  break;
	case 'p':
	  mode = CPP_MODE;
	  break;
	case 'c':
	  complete_file = 1;
	  break;
	case 'z':
	  center = 1;
	  break;
	case 'f':
	  frame = 1;
	  break;
	case 'o':
	  if (piped)
	    {
	      fprintf (stderr,
		       "%s: Can't use {-t,+pipe} and {-o,+output} together\n",
		       program_name);
	      exit(5);
	    }
	  output_name = optarg;
	  break;
	case 's':
	  font_size = optarg;
	  break;
	case 'l':
	  line_sep = optarg;
	  break;
	case 'i':
	  indentation = optarg;
	  break;
	case 'T':
	  tabtotab = atoi(optarg);
	  break;
        case 't':
	  if (output_name != 0)
	    {
	      fprintf (stderr,
		       "%s: Can't use {-t,+pipe} and {-o,+output} together\n",
		       program_name);
	      exit(5);
	    }
	  piped = 1;
	  break;
	case 'h':
	  header = 1;
	  complete_file = 1;	/* header implies complete-file */
	  break;
	case 'b':
	  line_begin = optarg;
	  break;
	case 'e':
	  line_end = optarg;
	  break;
	case 'u':
	  keyword_underline = 1;
	  break;
	case 'C':
	  comment_font = optarg;
	  break;
	case 'F':
	  base_font = optarg;
	  break;
	case 'N':
	  listing_name = optarg;
	  break;
	case 'L':
	  label_name = optarg;
	  break;
	case 'D':
	  default_label_name = optarg;
	  break;
	case 'R':
	  reference_name = optarg;
	  break;
	case 'H':
	  header_font = optarg;
	  break;
	case 'P':
	  cpp_font = optarg;
	  break;
	case 'S':
	  string_font = optarg;
	  break;
	case 'K':
	  keyword_font = optarg;
	  break;
	case 'V':
	  fprintf (stderr, "%s\n", version_string);
	  break;
        case 'x':
          want_index = 1;
          break;
	default:
	  usage(program_name);
	}
    }
  has_filename = (argc - optind == 1);
  if (has_filename)		/* last argument is input file name */
    {
      input_name = argv[optind];
      if (freopen (input_name, "r", stdin) == NULL)
	{
	  fprintf (stderr, "%s: Can't open `%s' for reading\n",
		   program_name, input_name);
	  exit (2);
	}
    }
  if ((output_name == 0) && !piped)
    {
      char * tmp;
      if (has_filename)
	{
	  char * point;
	  
	  point = strrchr (input_name, '/');
	  if (point == 0)	/* plain filename */
	    {
	      point = input_name;
	    }
	  else
	    {
	      point++;
	    }
	  tmp = malloc (strlen (point) + 1);
	  if (tmp == 0)
	    {
	      fprintf (stderr, "%s: Virtual memory exhausted\n", program_name);
	      exit (3);
	    }
	  strcpy (tmp, point);
	  point = strrchr (tmp, '.');
	  if (point != 0)
	    {
	      *point = '\0';
	    }
	}
      else
	{
	  tmp = program_name;
	}
      output_name = malloc (strlen (tmp) + 4 + 1);
      if (output_name == 0)
	{
	  fprintf (stderr, "%s: Virtual memory exhausted\n", program_name);
	  exit (3);
	}
      strcpy (output_name, tmp);
      strcat (output_name, ".tex");
    }
  if (!piped)
    {
      if (freopen (output_name, "w", stdout) == NULL)
	{
	  fprintf (stderr, "%s: Can't open `%s' for writing\n",
		   program_name, output_name);
	  exit (4);
	}
    }

  if (complete_file)
    {
      if (header)
	{
	  if (strcmp (font_size, "10pt") == 0)
	    {
	      printf ("\\documentclass{article}\n");
	      printf ("\\usepackage{fancyheadings}\n");
	    }
	  else
	    {
	      printf ("\\documentclass[%s]{article}\n",
		      font_size);
	      printf ("\\usepackage{fancyheadings}\n");
	    }
	}
      else
	{
	  if (strcmp (font_size, "10") == 0)
	    {
	      printf ("\\documentclass{article}\n");
	    }
	  else
	    {
	      printf ("\\documentclass[%s]{article}\n", font_size);
	    }
	}
      printf ("\\setlength{\\textwidth}{15cm}\n");
      printf ("\\setlength{\\textheight}{22.5cm}\n");
      printf ("\\setlength{\\hoffset}{-2cm}\n");
      printf ("\\setlength{\\voffset}{-2cm}\n");
      if (want_index)
        printf("\\makeindex\n");
      if (header)
	{
	  printf ("\\chead{%s Produced from ", header_font);
	  printf("%s", substitute(input_name));
	  printf (" at %.24s}\n", today);
	  printf ("\\cfoot{\\rm\\thepage}\n");
	  printf ("\\addtolength{\\headheight}{14pt}\n");
	  printf ("\\pagestyle{fancy}\n");
	}
      printf ("\\begin{document}\n");
    }

  printf ("\\ifx\\vhsListing\\undefined\\errmessage{You must include ");
  printf ("progdoc.sty}\\fi\n");
  printf ("\\expandafter\\ifx\\csname indentation\\endcsname\\relax\n");
  printf ("\\newlength{\\indentation}\\fi\n");

  printf ("{\n\\setlength{\\LTpre}{\\pdPre}\n");
  printf ("\\setlength{\\LTpost}{\\pdPost}\n");
  printf ("\\setlength{\\LTleft}{\\pdLeft}\n");
  printf ("\\setlength{\\LTright}{\\pdRight}\n");

  if (listing_name) {
    new_name = malloc(2*strlen(listing_name));
    temp = new_name;
    while (*listing_name) {
     if (*listing_name == '_') *new_name++ = '\\';
     *new_name++ = *listing_name++;
    }
    *new_name = '\0';
    listing_name = temp;
    printf ("\\sbox{\\ListingFirstHead}{\\small \\underline{{\\bfseries ");
    printf ("\\ListingName~\\theListing:} %s", listing_name);
    if (line_begin && line_end)
      printf("~[\\LineName~%s~\\toName~%s]", line_begin, line_end); 
    if (reference_name)
      printf("~(\\ReferenceName~Listing \\ref{%s} \\PageName~\\pageref{%s})",
	     reference_name, reference_name); 
    printf ("}}\n");
    printf ("\\sbox{\\ListingHead}{\\small \\underline{{\\bfseries \\ListingName~");
    printf ("\\theListing:} %s", listing_name);
    if (line_begin && line_end)
      printf("~[\\LineName~%s~\\toName~%s]", line_begin, line_end); 
    printf ("~(\\ListingContinue)}}\n");
  }
  printf ("\\usefont{OT1}{%s}{m}{n}\n", base_font);
  printf ("\\fontsize{%s}{%s}\\selectfont\n", font_size, line_sep);
  printf ("\\settowidth{\\indentation}{X}\n");
  printf ("\\renewcommand{\\{}{\\symbol{123}}\n");
  printf ("\\renewcommand{\\}}{\\symbol{125}}\n");
  printf ("\\setlength{\\tabcolsep}{0cm}\n");
  if (label_name) printf ("\\mylabel{%s}\n", label_name);
  if (center) {
    if(frame)
      printf ("\\begin{longtable}[c]{|l@{\\hspace{5mm}}|}\n");
    else
      printf ("\\begin{longtable}[c]{l}\n");
  }
  else { 
    if(frame)
      printf ("\\begin{longtable}{|l@{\\hspace{5mm}}|}\n");
    else
      printf ("\\begin{longtable}{l}\n");
  }
  if (listing_name) {
    printf ("\\multicolumn{1}{l}{\\usebox{\\ListingFirstHead}\\vspace{0.5cm}}\\\\");
  }
  if(frame) {
    printf ("\\hline\n\\\\\n");
  }
  printf ("\\endfirsthead\n");
  if (listing_name) { 
    printf ("\\multicolumn{1}{l}{\\usebox{\\ListingHead}\\vspace{0.5cm}}\\\\");
  }
  if(frame) {
    printf ("\\hline\n\\\\\n");
  }
  printf ("\\endhead\n");

  if(frame) {
    printf ("\\\\\n\\hline\n");
  }
  printf ("\\multicolumn{1}{r}{\\NextPage}\\endfoot\n");
  if(frame) {
    printf ("\n\\hline\n");
  }
  printf ("\\endlastfoot\n");
  yylex();
  printf ("\\addtocounter{table}{-1}\n");
  printf ("\\end{longtable}\n");
  printf ("}\n");

  if( !nohtml ) {
  
    printf("\\begin{htmlonly}\n");
    if (default_label_name) printf ("\\label{%s}\n", default_label_name);
    if (label_name) printf ("\\label{%s}\n", label_name);
    printf("\\end{htmlonly}\n");

    printf("\\begin{rawhtml}\n");
    printf("<TABLE BORDER=\"2\" CELLSPACING=\"0\" CELLPADDING=\"10\" ");
    printf("RULES=\"none\" >\n");
    printf("\\end{rawhtml}\n");

    printf("\\begin{htmlonly}\n");
    if (listing_name) {
      printf("\\HTML[STYLE=\"text-indent : 0cm; text-align : left; ");
      printf("text-decoration : underline\"]{CAPTION}{\\HTML{B}{\\ListingName~");
      printf("\\theListing:} %s", listing_name);
      if (line_begin && line_end)
        printf("~[\\LineName~%s~\\toName~%s]", line_begin, line_end); 
      if (reference_name)
        printf("~(\\ReferenceName~Listing \\ref{%s})", reference_name); 
      printf("}\n");
    }
    printf("\\end{htmlonly}\n");
    printf("\\begin{rawhtml}\n");
    printf("<TR>\n<TD STYLE=\"color : %s; ", html_color);
    printf("background-color : %s;\" >\n<PRE STYLE=\"margin-bottom : 0;\">",
	   html_background);

    fflush(html_file);
    rewind(html_file);
    while (fgets(buf, 512, html_file)) 
      printf("%s", buf);

    printf("</PRE>\n<TR>\n</TABLE>\n<BR>\n");
    printf("\\end{rawhtml}\n");
  }

  /* we try to keep the counter stuff outside the LaTeX and HTML conditionals*/

  if (listing_name) {
    printf ("\\stepcounter{Listing}\n");
  }
  if (complete_file) printf ("\\end{document}\n");

  exit(0);
}


void usage(const char * name)
{
  fprintf (stderr, "%s\n", version_string);
  fprintf (stderr, "\
Usage: %s [-a] [-c] [-h] [-i length] [-o path] [-p] [-s size] [-l leng]\n\
       [-t] [-F font] [-C font] [-H font] [-K font] [-P font] [-S font]\n\
       [-T wide] [-N filename] [-L label]  [-D label] [-R reference]\n\
       [-b line_nr] [-e line_nr] [-V] [-x] [-z] [-f] [-1] [-u]\n\
       [+ansi-c] [+complete-file] [+header] [+indentation length]\n\
       [+output path] [+c-plusplus] [+font-size size] [+line-sep length]\n\
       [+pipe] [+base-font font]\n\
       [+comment-font font] [+header-font font] [+cpp-font font]\n\
       [+keyword-font font] [+string-font font] [+tabstop wide]\n\
       [+listing-name filename] [+label-name name] [+def-label-name name]\n\
       [+referenced_by name] [+line-begin line_nr] [+line-end line_nr]\n\
       [+version] [+index] [+center] [+frame] [+path] [+plain-text]\n\
       [+keyword_underline]\n", name);
  exit (1);
}


int yywrap()
{
  return 1;
}


