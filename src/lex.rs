use std::io;
use std::ops::Range;
use std::path::Path;
use std::path::PathBuf;

use bumpalo::Bump;
use logos::Logos;

use crate::AllocPath;

#[derive(Debug, Clone, Copy)]
pub struct Token<'a> {
    pub kind: TokenKind,
    loc: Loc<'a>,
}

impl<'a> Token<'a> {
    pub(crate) fn loc(&self) -> Loc<'a> {
        self.loc
    }

    pub fn slice(&self) -> &'a str {
        &self.loc.src[self.loc.span()]
    }

    pub fn as_debug_string(&self) -> String {
        format!("{} {} {}", self.kind.kind_str(), self.slice(), self.value())
    }

    fn value(&self) -> TokenValue {
        match self.kind {
            TokenKind::String => TokenValue::String(&self.slice()[1..self.slice().len() - 1]),
            TokenKind::Number => TokenValue::Number(self.slice().parse().unwrap()),
            _ => TokenValue::None,
        }
    }
}

enum TokenValue<'a> {
    None,
    String(&'a str),
    Number(f64),
}

impl std::fmt::Display for TokenValue<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use TokenValue::*;
        match self {
            None => write!(f, "null"),
            String(s) => write!(f, "{s}"),
            Number(x) => write!(f, "{x:?}"),
        }
    }
}

#[derive(Clone, Copy)]
pub struct Loc<'a> {
    file: &'a Path,
    src: &'a str,
    span: Span,
}

impl std::fmt::Debug for Loc<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{:?}", self.file.display(), self.span())
    }
}

impl<'a> Loc<'a> {
    pub(crate) fn file(&self) -> &'a Path {
        self.file
    }

    pub(crate) fn start(&self) -> usize {
        self.span.start
    }

    pub(crate) fn span(&self) -> Range<usize> {
        self.span.start..self.span.end
    }

    pub(crate) fn until(self, other: Self) -> Self {
        assert_eq!(self.file, other.file);
        assert_eq!(self.src, other.src);
        assert!(self.span.end <= other.span.start);
        Self {
            span: Span {
                start: self.span.start,
                end: other.span.end,
            },
            ..self
        }
    }

    pub(crate) fn report(&self, kind: ariadne::ReportKind<'a>) -> ariadne::ReportBuilder<'a, Self> {
        ariadne::Report::build(kind, self.file(), self.start())
    }

    pub(crate) fn cache(&self) -> impl ariadne::Cache<Path> + 'a {
        struct Cache<'b>(&'b Path, ariadne::Source);
        impl ariadne::Cache<Path> for Cache<'_> {
            fn fetch(
                &mut self,
                id: &Path,
            ) -> Result<&ariadne::Source, Box<dyn std::fmt::Debug + '_>> {
                if self.0 == id {
                    Ok(&self.1)
                }
                else {
                    Err(Box::new(format!(
                        "failed to fetch source `{}`",
                        id.display(),
                    )))
                }
            }

            fn display<'a>(&self, id: &'a Path) -> Option<Box<dyn std::fmt::Display + 'a>> {
                Some(Box::new(id.display()))
            }
        }
        Cache(self.file(), ariadne::Source::from(self.src))
    }
}

impl ariadne::Span for Loc<'_> {
    type SourceId = Path;

    fn source(&self) -> &Self::SourceId {
        self.file
    }

    fn start(&self) -> usize {
        self.span.start
    }

    fn end(&self) -> usize {
        self.span.end
    }
}

#[derive(Debug, Clone, Copy)]
struct Span {
    start: usize,
    end: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Logos)]
#[logos(skip r"[ \n\r\t\f]+")]
#[logos(skip r"//[^\n]*\n?")]
pub enum TokenKind {
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,
    #[token(",")]
    Comma,
    #[token(".")]
    Dot,
    #[token("-")]
    Minus,
    #[token("+")]
    Plus,
    #[token(";")]
    Semicolon,
    #[token("/")]
    Slash,
    #[token("*")]
    Star,
    #[token("**")]
    StarStar,

    #[token("!")]
    Bang,
    #[token("!=")]
    BangEqual,
    #[token("=")]
    Equal,
    #[token("==")]
    EqualEqual,
    #[token(">")]
    Greater,
    #[token(">=")]
    GreaterEqual,
    #[token("<")]
    Less,
    #[token("<=")]
    LessEqual,

    #[regex(r"[\p{XID_start}_]\p{XID_continue}*")]
    Identifier,
    #[regex(r#""[^"]*""#)]
    String,
    #[regex(r"\d+(\.\d+)?")]
    Number,

    #[token("and")]
    And,
    #[token("class")]
    Class,
    #[token("else")]
    Else,
    #[token("false")]
    False,
    #[token("fun")]
    Fun,
    #[token("for")]
    For,
    #[token("if")]
    If,
    #[token("nil")]
    Nil,
    #[token("or")]
    Or,
    #[token("print")]
    Print,
    #[token("return")]
    Return,
    #[token("super")]
    Super,
    #[token("this")]
    This,
    #[token("true")]
    True,
    #[token("var")]
    Var,
    #[token("while")]
    While,
}

impl TokenKind {
    pub fn kind_str(self) -> String {
        use TokenKind::*;
        match self {
            LParen => "LEFT_PAREN".to_owned(),
            RParen => "RIGHT_PAREN".to_owned(),
            LBrace => "LEFT_BRACE".to_owned(),
            RBrace => "RIGHT_BRACE".to_owned(),
            LBracket => "LEFT_BRACKET".to_owned(),
            RBracket => "RIGHT_BRACKET".to_owned(),
            BangEqual => "BANG_EQUAL".to_owned(),
            EqualEqual => "EQUAL_EQUAL".to_owned(),
            GreaterEqual => "GREATER_EQUAL".to_owned(),
            LessEqual => "LESS_EQUAL".to_owned(),
            _ => format!("{self:?}").to_uppercase(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error<'a> {
    #[error("lexer error at {at:?}")]
    LexerError { at: crate::lex::Loc<'a> },
    #[error("could not read file `{filename}`")]
    IoError {
        filename: PathBuf,
        #[backtrace]
        source: io::Error,
    },
}

pub fn lex<'a>(
    bump: &'a Bump,
    filename: impl AsRef<Path>,
    src: &str,
) -> Result<&'a [Token<'a>], Error<'a>> {
    let filename = bump.alloc_path(filename.as_ref());
    let src = bump.alloc_str(src);

    let tokens = TokenKind::lexer(src)
        .spanned()
        .map(|(kind, span)| {
            let loc = Loc {
                file: filename,
                src,
                span: Span { start: span.start, end: span.end },
            };
            Ok(Token {
                kind: kind.map_err(|()| Error::LexerError { at: loc })?,
                loc,
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;

    Ok(bump.alloc_slice_copy(&tokens))
}
