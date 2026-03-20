% author: aqnin@outlook.com
% reference standard:
%   http://jwc.nankai.edu.cn/2022/1124/c24a497818/page.htm
%   https://github.com/Tr0py/NKU-thesis-template-2020/files/11620969/2023V2.pdf

\NeedsTeXFormat{LaTeX2e}[2021/11/15]
\ProvidesClass{nkuthesis}[2023/08/30 Nankai University bachelor's thesis]

\PassOptionsToPackage{quiet}{fontspec} % suppress warning

\DeclareOption{draft}{\PassOptionsToClass{\CurrentOption}{ctexart}}
\ProcessOptions\relax
\LoadClass[zihao=-4,linespread=1.64]{ctexart}

\RequirePackage{etoolbox}
\RequirePackage{mathtools}
\RequirePackage{geometry}
\RequirePackage{enumitem}
\RequirePackage{pifont}
\RequirePackage[perpage,bottom]{footmisc}
\RequirePackage{caption}
\RequirePackage{titletoc}
\RequirePackage{fancyhdr}
\RequirePackage{longtable}
\RequirePackage{listings}
\RequirePackage[dvipsnames]{xcolor}
\RequirePackage[ruled]{algorithm}
\RequirePackage{flafter}
\RequirePackage[section]{placeins}
\RequirePackage{chngcntr}
\RequirePackage[hidelinks,pdfa,pdfusetitle,bookmarksnumbered]{hyperref}
\RequirePackage{titlesec}
\RequirePackage{mfirstuc-english}
\RequirePackage[gbpunctin=false,gbnamefmt=lowercase,style=gb7714-2015]{biblatex}


\pdfstringdefDisableCommands{\let\\\empty} % suppress warning

%% font
\setmainfont{Times New Roman}
\setsansfont{Times New Roman Bold}

%% margin
\geometry{
  paper=a4paper,
  vmargin=2.54cm,
  hmargin=3.17cm,
  headheight=0.44cm,
  headsep=0.6cm,
  footskip=0.79cm,
}

%% page header
\fancypagestyle{chdr}{
  \fancyhf{}
  \fancyhead[C]{\zihao{-5}\leftmark}
  \fancyfoot[C]{\zihao{-5}\thepage}
  \renewcommand{\headrulewidth}{0.4pt}
  \renewcommand{\headruleskip}{1pt}
}
\pagestyle{chdr}

%% title
\AddToHook{cmd/section/before}{\newpage} % 章另启一页
\ctexset{
  secnumdepth=subsubsection,
  section = {
    name={第,章},
    number=\chinese{section},
    format=\centering\zihao{3}\sffamily,
  },
  subsection = {
    name={第,节},
    number=\chinese{subsection},
    format=\centering\zihao{-3}\sffamily,
  },
  subsubsection = {
    number=\arabic{section}.\arabic{subsection}.\arabic{subsubsection},
    format=\raggedright\zihao{4}\sffamily,
  },
}

%% footnote
\renewcommand{\thefootnote}{\ding{\numexpr171+\value{footnote}}}
\NewCommandCopy\old@makefntext{\@makefntext}
\renewcommand{\@makefntext}[1]{\old@makefntext{\enspace{}#1}}

%% caption
\DeclareCaptionLabelSeparator{enspace}{\enspace}
\captionsetup{
  labelsep=enspace,
  skip=0pt,
  font=small,
}
\captionsetup[ruled]{font=small}

%% list
\setlist{nosep}

%% table
\AddToHook{cmd/caption/before}{\linespread{1.5}\zihao{5}}
\AddToHook{env/longtable/begin}{\linespread{1.5}\zihao{5}}

%% code
\renewcommand{\lstlistingname}{代码}
\lstset{
	frame=lines,
    breaklines,
    basicstyle=\ttfamily\zihao{5}\linespread{1}\selectfont,
	keywordstyle=\bfseries\color{NavyBlue},
	emphstyle={\bfseries\color{Rhodamine}},
	commentstyle=\itshape\color{black!50!white},
	aboveskip=10pt,
	columns=flexible,
	tabsize=4,
}

%% algorithm
\floatname{algorithm}{算法}
\counterwithin{algorithm}{section}
\AddToHook{env/algorithm/begin}{\linespread{1}\zihao{5}}

%% number
\counterwithin{table}{section}
\counterwithin{figure}{section}
\counterwithin{equation}{section}
\AtBeginDocument{\counterwithin{lstlisting}{section}}

%% label
\labelformat{section}{第\zhnumber{#1}章}
\renewcommand{\thesubsection}{\chinese{subsection}}
\labelformat{subsection}{第#1节}
\renewcommand{\thesubsubsection}{\arabic{section}.\arabic{subsection}.\arabic{subsubsection}}
\labelformat{subsubsection}{\space{}#1~小节}
\labelformat{figure}{图~#1\space}
\labelformat{table}{表~#1\space}
\labelformat{equation}{式~(#1)\space}
\labelformat{algorithm}{算法~#1\space}
\AtBeginDocument{\labelformat{lstlisting}{代码~#1\space}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Custom Macro %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% fakebold
\newenvironment{fakebold}[1]
  {\special{pdf:code q 2 Tr #1 w}}
  {\special{pdf:code Q}}
\NewDocumentCommand{\textfb}{O{1}m}
  {\begin{fakebold}{#1}\textnormal{#2}\end{fakebold}}

%% unnumbered title
\NewDocumentCommand{\nolabel@section}{smO{#4}m}
  {\section*{\phantomsection\centering\sffamily\zihao{#2}#4}
    \markboth{\centering{}#3}{}
    \IfBooleanF{#1}{\addcontentsline{toc}{section}{#3}}}

%% title box
\DeclareListParser{\dolinelist}{\\}
\newcommand{\uline@parbox}[2]{%
  \renewcommand*{\do}[1]{\underbar{\makebox[#1][l]{##1}}\\}
  \begin{tabular}[t]{@{}l@{}}
  \expandafter\dolinelist\expandafter{#2}
  \end{tabular}%
}

%% cover
\newcommand{\zh@title}{}
\newcommand{\en@title}{}
\newcommand{\@studentid}{}
\newcommand{\@grade}{}
\newcommand{\@major}{}
\newcommand{\@department}{}
\newcommand{\@college}{}
\newcommand{\@adviser}{}
\newcommand{\studentid}[1]{\renewcommand{\@studentid}{#1}}
\newcommand{\grade}[1]{\renewcommand{\@grade}{#1}}
\newcommand{\major}[1]{\renewcommand{\@major}{#1}}
\newcommand{\department}[1]{\renewcommand{\@department}{#1}}
\newcommand{\college}[1]{\renewcommand{\@college}{#1}}
\newcommand{\adviser}[1]{\renewcommand{\@adviser}{#1}}
\RenewDocumentCommand{\title}{sm}{\IfBooleanTF{#1}
  {\renewcommand{\en@title}{#2}}
  {\renewcommand{\zh@title}{#2}}}
\renewcommand{\maketitle}{%
  \hypersetup{pdftitle={\zh@title}}
  \hypersetup{pdfsubject={南开大学本科生毕业论文（设计）}}
  
  \begin{titlepage}
    \hbadness=10000
    \centering
      \vspace*{25pt}\par\linespread{1}\fontsize{56}{0}
    \textfb{\makebox[5.5em][s]{南开大学}}\par
      \vspace{35pt}\par\linespread{1}\zihao{2}
    \makebox[255pt][s]{本科生毕业论文（设计）}\par
      \vspace{60pt}\par\linespread{1.2}\zihao{3}
    \begin{tabular}
      {p{4em}@{：}p{\widthof{
        \begin{tabular}{@{}l@{}}\zh@title\\\en@title\end{tabular}}}}
      \makebox[\hsize][s]{中文题目} &
      {\kaishu\mbox{\fakebold{0.2}\uline@parbox{\hsize}{\zh@title}\endfakebold}}
       \\
      \makebox[\hsize][s]{外文题目} &
      \mbox{\uline@parbox{\hsize}{\en@title}} \\
    \end{tabular}
      \vspace{\fill}\par\linespread{1.15}\zihao{3}
    \begin{tabular}
      {p{4em}@{：}p{\widthof{
        \begin{tabular}{@{}l@{}}\@studentid\\\@author\\\@grade\\\@major\\
          \@department\\\@college\\\@adviser\\\@date\end{tabular}}}}
      \makebox[\hsize][s]{学号} & {\kaishu\uline@parbox{\hsize}{\@studentid}} \\
      \makebox[\hsize][s]{姓名} & {\kaishu\uline@parbox{\hsize}{\@author}} \\
      \makebox[\hsize][s]{年级} & {\kaishu\uline@parbox{\hsize}{\@grade}} \\
      \makebox[\hsize][s]{专业} & {\kaishu\uline@parbox{\hsize}{\@major}} \\
      \makebox[\hsize][s]{系别} & {\kaishu\uline@parbox{\hsize}{\@department}} \\
      \makebox[\hsize][s]{学院} & {\kaishu\uline@parbox{\hsize}{\@college}} \\
      \makebox[\hsize][s]{指导教师} & {\kaishu\uline@parbox{\hsize}{\@adviser}} \\
      \makebox[\hsize][s]{完成日期} & {\kaishu\uline@parbox{\hsize}{\@date}} \\
    \end{tabular}
      \vspace{\fill}
  \end{titlepage}
  \begin{titlepage}
    \nolabel@section*{3}[声明]{关于南开大学本科生毕业论文（设计）的声明}
    本人郑重声明：所呈交的学位论文，是本人在指导教师指导下，进行研究工作所取得的成果。
    除文中已经注明引用的内容外，本学位论文的研究成果不包含任何他人创作的、
    已公开发表或没有公开发表的作品内容。对本论文所涉及的研究工作做出贡献的其他
    个人和集体，均已在文中以明确方式标明。本学位论文原创性声明的法律责任由本人承担。
    \par\bigskip\begin{flushright}
      学位论文作者签名：\hspace*{6em}\medskip\par
      年\hspace{2em}月\hspace{2em}日
    \end{flushright}\vspace{4\baselineskip}\par
    本人声明：该学位论文是本人指导学生完成的研究成果，已经审阅过论文的全部内容，
    并能够保证题目、关键词、摘要部分中英文内容的一致性和准确性。
    \par\bigskip\begin{flushright}
      学位论文指导教师签名：\hspace*{6em}\medskip\par
      年\hspace{2em}月\hspace{2em}日
    \end{flushright}
    \end{titlepage}
}

%% keywords
\newcommand{\zh@keywords}{}
\newcommand{\en@keywords}{}
\NewDocumentCommand{\keywords}{sm}
  {\IfBooleanTF{#1}
    {\renewcommand{\en@keywords}{#2}}
    {\renewcommand{\zh@keywords}{#2}\hypersetup{pdfkeywords={#2}}}}

%% chinese abstract
\renewenvironment{abstract}
  {\clearpage\nolabel@section{4}[摘要]{摘\hspace{1em}要}}
  {\par\bigskip\par
    \hangafter 1 \hangindent 4em \noindent
    \mbox{\textsf{关键词：}}\zh@keywords
    \clearpage}

%% english abstract
\newenvironment{abstract*}
  {\clearpage\nolabel@section{4}{Abstract}}
  {\par\bigskip\par
    \hangafter 1 \hangindent 5.5em \noindent
    \mbox{\textsf{Key Words:\space}}\en@keywords
    \clearpage}

%% table of content
\setcounter{tocdepth}{3}
\titlecontents{section}[0em]{\zihao{-3}}
  {\thecontentslabel\quad}{}{\titlerule*{.}\contentspage}
\titlecontents{subsection}[2em]{\zihao{4}}
  {\thecontentslabel\quad}{}{\titlerule*{.}\contentspage}
\titlecontents{subsubsection}[4em]{\zihao{-4}}
  {\thecontentslabel\quad}{}{\titlerule*{.}\contentspage}
\renewcommand{\tableofcontents}{%
  \clearpage
  \nolabel@section{3}[目录]{目\hspace{1em}录}
  {\linespread{1.2}\@starttoc{toc}}
  \clearpage
}

%% 附录
\AddToHook{cmd/appendix/after}{
\clearpage
\nolabel@section{4}[附录]{附\hspace{1em}录}
}

%% bibliography
\DefineBibliographyStrings{english}{
  andothers  = {\textit{et al.}},
  in={},
  incn={},
}
\DeclareFieldFormat{titlecase}{\MakeSentenceCase*{#1}}
\MFUnocap{of}\MFUnocap{to}\MFUnocap{as}\MFUnocap{by}\MFUnocap{on}\MFUnocap{in}
\MFUnocap{about}\MFUnocap{from}\MFUnocap{with}\MFUnocap{for}\MFUnocap{at}
\MFUnocap{up}\MFUnocap{down}\MFUnocap{under}\MFUnocap{over}\MFUnocap{against}
\DeclareFieldFormat{journaltitle}{\capitalisewords{#1}\isdot}
\DeclareFieldFormat{booktitle}{\capitalisewords{#1}}
\setlength{\bibitemsep}{0pt}
\renewcommand{\finentrypunct}{}
\renewcommand{\bibfont}{\zihao{5}\linespread{1.2}}
\defbibheading{bibliography}[\bibname]{\clearpage\nolabel@section{4}{#1}}

%% acknowledgement
\newenvironment{acknowledgement}
  {\clearpage\nolabel@section{4}[致谢]{致\hspace{1em}谢}}
  {\clearpage}
