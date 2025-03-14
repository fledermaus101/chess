use std::fmt::Display;

use crate::square::Square;

#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub enum PieceType {
    King,
    Queen,
    Rook,
    Bishop,
    Knight,
    Pawn,
}

impl Display for PieceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::King => "King",
            Self::Queen => "Queen",
            Self::Rook => "Rook",
            Self::Bishop => "Bishop",
            Self::Knight => "Knight",
            Self::Pawn => "Pawn",
        };
        write!(f, "{name}")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub struct Piece {
    square: Square,
    piece_type: PieceType,
    is_white: bool,
}

impl Display for Piece {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut piece_name = {
            match self.piece_type() {
                PieceType::King => "K",
                PieceType::Queen => "Q",
                PieceType::Rook => "R",
                PieceType::Bishop => "B",
                PieceType::Knight => "N",
                PieceType::Pawn => "",
            }
        }
        .to_string();
        if !self.is_white() {
            piece_name.make_ascii_lowercase();
        }
        write!(f, "{}{}", piece_name, self.square())
    }
}

impl Piece {
    #[must_use]
    pub const fn square(&self) -> Square {
        self.square
    }

    #[must_use]
    pub const fn piece_type(&self) -> PieceType {
        self.piece_type
    }

    #[must_use]
    pub const fn new(square: Square, piece_type: PieceType, is_white: bool) -> Self {
        Self {
            square,
            piece_type,
            is_white,
        }
    }

    #[must_use]
    pub const fn is_white(&self) -> bool {
        self.is_white
    }

    #[must_use]
    pub fn legal_moves(&self) -> &[Square] {
        match self.piece_type() {
            PieceType::King => todo!(),
            PieceType::Queen => todo!(),
            PieceType::Rook => todo!(),
            PieceType::Bishop => todo!(),
            PieceType::Knight => todo!(),
            PieceType::Pawn => todo!(),
        }
    }

    #[must_use]
    pub const fn piece_letter(&self) -> char {
        let name = match self.piece_type() {
            PieceType::King => 'K',
            PieceType::Queen => 'Q',
            PieceType::Rook => 'R',
            PieceType::Bishop => 'B',
            PieceType::Knight => 'N',
            PieceType::Pawn => 'P',
        };

        if self.is_white {
            name
        } else {
            name.to_ascii_lowercase()
        }
    }
}
