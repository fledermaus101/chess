use crate::{
    piece::{Piece, PieceType},
    square::Square,
};
use std::fmt::Debug;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Move {
    from: Square,
    to: Square,
    piece_type: PieceType,
    is_white: bool,
    promotion_piece: Option<PieceType>,
    is_castling: CastleMove,
}

impl std::fmt::Display for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <bool as std::fmt::Display>::fmt(&self.is_white, f)
    }
}

impl Move {
    #[must_use]
    pub const fn new(
        from: Square,
        to: Square,
        piece_type: PieceType,
        is_white: bool,
        promotion_piece: Option<PieceType>,
        is_castling: CastleMove,
    ) -> Self {
        Self {
            from,
            to,
            piece_type,
            is_white,
            promotion_piece,
            is_castling,
        }
    }

    pub fn from(&self) -> Square {
        self.from
    }

    pub fn to(&self) -> Square {
        self.to
    }

    pub fn piece_type(&self) -> PieceType {
        self.piece_type
    }

    pub fn is_white(&self) -> bool {
        self.is_white
    }

    pub fn promotion_piece(&self) -> Option<PieceType> {
        self.promotion_piece
    }

    pub fn is_castling(&self) -> CastleMove {
        self.is_castling
    }
}

impl Debug for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Move {{{} -> {}, promotion_piece: {:?}, is_castling: {:?}}}",
            Piece::new(self.from, self.piece_type, self.is_white),
            self.to,
            self.promotion_piece,
            self.is_castling
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CastleMove {
    KingSide,
    QueenSide,
    None,
}
