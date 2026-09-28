% Jan Groenewald jan@aims.ac.za
% Don't change anything here
\NeedsTeXFormat{LaTeX2e}[1994/06/01]
\ProvidesClass{aimsessay}[2011/04/05 Class for AIMS Essays]
\LoadClass[11pt,a4paper]{report}

% Formatting and font
\usepackage[text={6.5in,9in}]{geometry}
\parskip=0.5\baselineskip
\parindent=0pt
\renewcommand{\familydefault}{\sfdefault}
%I prefer raggedright, but most people prefer justified.
%\raggedright
\raggedbottom % should be default already?

% AMS packages.
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{amsfonts}
\RequirePackage{amsthm}
\RequirePackage{mathtools}
\RequirePackage{latexsym}

% An additional colour for citecolor= in the hyperref pacakage
\RequirePackage{color}
\definecolor{brown}{rgb}{0.5,0.3,0.1}
\RequirePackage[pdftex,%
    colorlinks=true,urlcolor=blue,citecolor=brown]{hyperref}
\urlstyle{same}

\RequirePackage[pdftex]{graphicx}

% fix quotation marks like "this"
% NOTE incompatible with quiver labels
\RequirePackage{csquotes}
\MakeOuterQuote{"}

% Do not show "Chapter" each time
\def\@chapapp{}
% define chapter heading layout
\def\@makechapterhead#1{{%
    \parindent0pt\raggedright\huge\bf
    \@chapapp \thechapter.\hskip15pt #1\vskip15pt}}
% define chapter heading layout (when not numbered) make**S**chapterhead
\def\@makeschapterhead#1{{%
    \parindent0pt\raggedright\huge\bf
    \@chapapp #1\vskip15pt}}
\def\@makesectionhead#1{{%
    \parindent0pt\raggedright\large\bf
    \@chapapp \thechapter\hskip15pt #1\vskip15pt}}

% Running headers
%\headheight12pt
\def\ps@myheadings{%
  \let\@mkboth\markboth\def\@oddfoot{}\def\@evenfoot{}%
  \def\@evenhead{%
    \underline{\strut\hbox to\hsize{Page \thepage\hss\leftmark}}}
  \def\@oddhead{%
    \underline{\strut\hbox to\hsize{\rightmark\hss Page \thepage}}}
  \def\chaptermark##1{%
    \markboth{{Chapter \@chapapp\thechapter.\ ##1}}{}}
  \def\sectionmark##1{%
    \markright {\ifnum \c@secnumdepth >\z@ Section \thesection.\ ##1\fi }}%
}

% Make "\subsection" behave like the top of a theorem (in terms of spacing).
%\renewcommand{\subsection}{\@ifstar{\aims@subsection}{\aims@subsection}}
%\newcommand{\aims@subsection}[1]{%
%  \@startsection{subsection}{2}{\z@}{\z@}{\z@}%
%  {\normalfont\normalsize\bfseries}{\unskip~#1.~\hfill\break}}%\hfill\break}

%
% Undefine "\subsubsection", "\paragraph", "\subparagraph" in a nice way.
\newcommand{\aims@undefined@error}[1]{%
  \ClassError{aimsessay}{The command \string#1 is disabled in AIMS Essays}%
  {Please choose a less intricate structure for your Essay!}}
\renewcommand{\subsubsection}{\aims@undefined@error{\subsubsection}}
\renewcommand{\paragraph}    {\aims@undefined@error{\paragraph}}
\renewcommand{\subparagraph} {\aims@undefined@error{\subparagraph}}

% Produce an error when \\ is used in normal text 
\AtBeginDocument{\def\\{\ClassError{aimsessay}{%
      You should not use \protect\\ in normal text. If you want
      \MessageBreak
      to separate two paragraphs, just leave an empty line
      \MessageBreak
      in your code. Otherwise, why would you want a new
      \MessageBreak
      line? Basically, only use \protect\\ in your title, in
      \MessageBreak
      tables, in matrices, and in aligned equations.}{}}}

% Do not show subsections and theorems in Table of Contents
\AtBeginDocument{\setcounter{tocdepth}{1}}

% Some few people still use appendices. We discourage them. To use it, make a normal chapter, but put \appendix in the first line and \endappendix in the last line.
\def\appendix{\setcounter{chapter}{0}\setcounter{section}{0}\def\thechapter{\Alph{chapter}}\def\@chapapp{Appendix }}
\def\endappendix{\setcounter{chapter}{0}\setcounter{section}{0}\def\thechapter{\Alph{chapter}}\def\@chapapp{}}

