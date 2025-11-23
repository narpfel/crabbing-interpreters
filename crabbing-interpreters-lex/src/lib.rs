#![feature(type_alias_impl_trait)]
#![feature(closure_lifetime_binder)]

use std::ops::Range;
use std::path::Path;

use ariadne::Config;
use ariadne::IndexType;
use bumpalo::Bump;
use logos::Logos;

#[derive(Clone, Copy)]
pub struct Token<'a> {
    pub kind: TokenKind,
    loc: Loc<'a>,
}

impl<'a> Token<'a> {
    pub fn loc(&self) -> Loc<'a> {
        self.loc
    }

    pub fn slice(&self) -> &'a str {
        self.loc.slice()
    }

    pub fn as_debug_string(&self) -> String {
        format!("{} {} {}", self.kind.kind_str(), self.slice(), self.value())
    }

    fn value(&self) -> TokenValue<'_> {
        match self.kind {
            TokenKind::String => TokenValue::String(&self.slice()[1..self.slice().len() - 1]),
            TokenKind::Number => TokenValue::Number(self.slice().parse().unwrap()),
            _ => TokenValue::None,
        }
    }
}

impl std::fmt::Debug for Token<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}({:?})@{:?}", self.kind, self.slice(), self.loc)
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
struct SourceFile<'a> {
    file: &'a Path,
    src: &'a str,
}

impl<'a> SourceFile<'a> {
    fn debug_source_file(bump: &'a Bump, src: &'a str) -> &'a Self {
        let file: &'static Path = Path::new("");
        bump.alloc(Self { file, src })
    }
}

#[derive(Clone, Copy)]
pub struct Loc<'a> {
    span: Span,
    source_file: &'a SourceFile<'a>,
}

impl<'a> Loc<'a> {
    pub fn debug_loc(bump: &'a Bump, src: &'a str) -> Self {
        Loc {
            span: Span { start: 0, end: src.len() },
            source_file: SourceFile::debug_source_file(bump, src),
        }
    }
}

impl std::fmt::Debug for Loc<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{:?}", self.file().display(), self.span())
    }
}

impl<'a> Loc<'a> {
    pub fn loc(&self) -> Self {
        *self
    }

    pub(crate) fn file(&self) -> &'a Path {
        self.source_file.file
    }

    fn src(&self) -> &'a str {
        self.source_file.src
    }

    pub(crate) fn span(&self) -> Range<usize> {
        self.span.start..self.span.end
    }

    pub fn slice(&self) -> &'a str {
        &self.src()[self.span()]
    }

    pub fn until(self, other: Self) -> Self {
        assert_eq!(self.file(), other.file());
        assert_eq!(self.src(), other.src());
        assert!(self.span.end <= other.span.start);
        Self {
            span: Span {
                start: self.span.start,
                end: other.span.end,
            },
            ..self
        }
    }

    pub fn report(&self, kind: ariadne::ReportKind<'a>) -> ariadne::ReportBuilder<'a, Self> {
        ariadne::Report::build(kind, *self).with_config(
            Config::default()
                .with_index_type(IndexType::Byte)
                .with_minimise_crossings(true),
        )
    }

    pub fn cache(&self) -> impl ariadne::Cache<Path> + 'a {
        struct Cache<'b>(&'b Path, ariadne::Source<&'b str>);

        impl<'b> ariadne::Cache<Path> for Cache<'b> {
            type Storage = &'b str;

            fn fetch(
                &mut self,
                id: &Path,
            ) -> Result<&ariadne::Source<&'b str>, impl std::fmt::Debug> {
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

            fn display<'a>(&self, id: &'a Path) -> Option<impl std::fmt::Display + 'a> {
                Some(Box::new(id.display()))
            }
        }

        Cache(self.file(), ariadne::Source::from(self.src()))
    }
}

impl ariadne::Span for Loc<'_> {
    type SourceId = Path;

    fn source(&self) -> &Self::SourceId {
        self.file()
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
    #[regex(r"[0-9]+(\.[0-9]+)?")]
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

#[derive(Debug, Clone, Copy)]
pub struct Error<'a> {
    pub at: Loc<'a>,
}

impl<'a> Error<'a> {
    pub fn loc(&self) -> Loc<'a> {
        self.at
    }
}

pub type TokenIter<'a> = impl Iterator<Item = Result<Token<'a>, Error<'a>>>;

#[define_opaque(TokenIter)]
pub fn lex<'a>(bump: &'a Bump, filename: &'a Path, src: &str) -> (TokenIter<'a>, Loc<'a>) {
    let src = bump.alloc_str(src);
    let source_file = &*bump.alloc(SourceFile { file: filename, src });
    let tokens = TokenKind::lexer(src).spanned().map(|(kind, span)| {
        let loc = Loc {
            span: Span { start: span.start, end: span.end },
            source_file,
        };
        Ok(Token {
            kind: kind.map_err(|()| Error { at: loc })?,
            loc,
        })
    });
    let eof_loc = Loc {
        span: Span { start: src.len(), end: src.len() },
        source_file,
    };
    (tokens, eof_loc)
}

#[cfg(test)]
mod test {
    use rstest::fixture;
    use rstest::rstest;

    use super::*;

    #[fixture]
    fn bump() -> Bump {
        Bump::new()
    }

    macro_rules! check {
        ($body:expr) => {
            for<'a> |result: TokenIter<'a>| -> () {
                #[allow(clippy::redundant_closure_call)]
                let () = $body(result.collect::<Result<Vec<_>, _>>());
            }
        };
    }

    macro_rules! check_err {
        ($pattern:pat) => {
            check!(|result| pretty_assertions::assert_matches!(result, Err($pattern)))
        };
    }

    const FULLWIDTH_NUMBER_4_LEN: usize = '４'.len_utf8();

    #[rstest]
    #[case::unicode_number(
        "４２",
        check_err!(Error { at: Loc { span: Span { start: 0, end: FULLWIDTH_NUMBER_4_LEN }, .. } }),
    )]
    fn test_lex_error(
        bump: Bump,
        #[case] src: &str,
        #[case] expected: impl for<'a> FnOnce(TokenIter<'a>),
    ) {
        expected(lex(&bump, Path::new("<src>"), src).0)
    }
}
