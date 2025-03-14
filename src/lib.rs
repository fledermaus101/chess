#![feature(iter_collect_into)]
#![feature(array_chunks)]
pub mod board;
pub mod r#move;
pub mod piece;
pub mod square;
pub mod squarelist;

use piece::PieceType;

/// Returns an index, given a [`PieceType`] and its `color`, commonly used as an index into an array
#[must_use]
pub const fn convert_piece_to_index(piece_type: PieceType, is_white: bool) -> usize {
    // 4 bits | 2 ^ 4 = 16 possible states
    // 12 are valid
    //                -> +6 (0110)
    // king   w : 0000 | b : 0110
    // queen  w : 0001 | b : 0111
    // rook   w : 0010 | b : 1000
    // bishop w : 0011 | b : 1001
    // knight w : 0100 | b : 1010
    // pawn   w : 0101 | b : 1011
    piece_type as usize + !is_white as usize * 6
}

/// Returns the [`PieceType`] of the index
///
/// # Panics
///
/// Panics if index > 11
#[must_use]
pub const fn get_index_piece(index: usize) -> PieceType {
    assert!(index <= 0b1011);
    PIECE_TYPE_VARIANTS[index % 6]
}

/// Returns the color of the index
///
/// # Panics
///
/// Panics if index > 11
#[must_use]
pub const fn get_index_color(index: usize) -> bool {
    assert!(index <= 0b1011);
    index < 0b0110
}

const PIECE_TYPE_VARIANTS: [PieceType; 6] = [
    PieceType::King,
    PieceType::Queen,
    PieceType::Rook,
    PieceType::Bishop,
    PieceType::Knight,
    PieceType::Pawn,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn piece_and_color_to_index() {
        assert_eq!(convert_piece_to_index(PieceType::Pawn, true), 0b0101);
        assert_eq!(
            convert_piece_to_index(PieceType::Pawn, false),
            0b0101 + 0b0110
        );
    }

    #[test]
    fn index_to_color() {
        let index = convert_piece_to_index(PieceType::Pawn, false);
        assert!(!get_index_color(index));
        assert!(get_index_color((index + 0b0110) % 0b0110));
    }

    #[test]
    fn index_to_piece() {
        for piece_type in PIECE_TYPE_VARIANTS {
            let index = convert_piece_to_index(piece_type, false);
            assert_eq!(get_index_piece(index), piece_type);
        }
    }
}
