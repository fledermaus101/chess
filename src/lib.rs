#![feature(maybe_uninit_uninit_array)]
#![feature(const_maybe_uninit_uninit_array)]
#![feature(maybe_uninit_slice)]
mod piecelist;

use crate::piecelist::SquareList;
use std::{fmt::Display, num::ParseIntError};

use bevy::prelude::{Component, Resource};
use thiserror::Error;

#[allow(unused)]
#[derive(Clone, Copy, Resource)]
pub struct Board {
    bit_boards: [u64; 12],
    squarelists: [SquareList; PIECE_TYPE_VARIANTS.len() * 2],
    side_to_move: bool,
    half_moves: u8, // should never be larger than 100, as that would be a draw
    full_moves: u32,
    en_passant_square: Option<Square>,
    castling_white_kingside: bool,
    castling_white_queenside: bool,
    castling_black_kingside: bool,
    castling_black_queenside: bool,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub struct Square(u8);

#[derive(Clone, Eq, PartialEq, Debug, Error)]
pub enum AlgebraicSqaureConversionError {
    #[error("String length was expected to be 2. Got {0}")]
    WrongLength(usize),
    #[error("Expected file char to be one of a, b, c, d, e, f, g, h. Got '{0}'")]
    InvalidFileChar(char),
    #[error("Expected rank char to be a digit. Got '{0}'")]
    RankCharIsNotADigit(char),
    #[error("Expected rank char to be a digit in the inclusive range from 1 to 8. Got {0}")]
    RankDigitIsOutsideOfValidRange(u32),
}

impl Display for Square {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", (self.file() + b'a') as char, self.rank() + 1)
    }
}

impl Square {
    pub const fn from_square(square_index: u8) -> Square {
        assert!(
            square_index <= 63,
            "square_index cannot be larger than 63 (0 indexed)"
        );
        Square(square_index)
    }

    pub const fn from_lateral(file: u8, rank: u8) -> Square {
        assert!(file <= 7, "file cannot be larger than 7 (0 indexed)");
        assert!(rank <= 7, "rank cannot be larger than 7 (0 indexed)");
        Square(rank * 8 + file)
    }

    pub const fn file(self) -> u8 {
        self.0 % 8
    }

    pub const fn rank(self) -> u8 {
        self.0 / 8
    }

    pub fn from_algebraic(square: &str) -> Square {
        Self::try_from_algebraic(square).expect("Square was not able to be parsed.")
    }

    pub fn try_from_algebraic(square: &str) -> Result<Square, AlgebraicSqaureConversionError> {
        if square.len() != 2 {
            return Err(AlgebraicSqaureConversionError::WrongLength(square.len()));
        }

        let mut chars = square.chars();
        let file = match chars.next().expect(
            "String square is empty. This should be impossible because of the length check.",
        ) {
            ch @ 'a'..='h' => ch as u8 - b'a',
            ch => return Err(AlgebraicSqaureConversionError::InvalidFileChar(ch)),
        };

        let rank = chars.next().expect("String is only one char long. This should be impossible because of the length check").to_digit(10).expect("Expected rank to be a digit");
        if !(1..=8).contains(&rank) {
            return Err(AlgebraicSqaureConversionError::RankDigitIsOutsideOfValidRange(rank));
        }
        Ok(Self::from_lateral(file, (rank - 1) as u8))
    }
}

#[allow(unused)]
impl Board {
    fn set_square(&mut self, piece: Piece) {
        let index = convert_piece_to_index(piece.piece_type(), piece.is_white());
        self.bit_boards[index] |= 1 << piece.square().0;

        self.squarelists[index].add(piece.square());
    }

    fn clear_square_of_piece(&mut self, piece: Piece) {
        let index = convert_piece_to_index(piece.piece_type(), piece.is_white());
        self.bit_boards[index] &= !(1 << piece.square().0);

        self.squarelists[index].remove(piece.square());
    }

    fn clear_square(&mut self, square: Square) {
        for bit_board in self.bit_boards.iter_mut() {
            *bit_board &= !(1 << square.0);
        }
        for squarelist in self.squarelists.iter_mut() {
            squarelist.remove(square);
        }
    }

    pub const fn get_bitboard(&self, piece_type: PieceType, is_white: bool) -> u64 {
        self.bit_boards[convert_piece_to_index(piece_type, is_white)]
    }

    pub const fn get_piecelist(&self, piece_type: PieceType, is_white: bool) -> SquareList {
        self.squarelists[convert_piece_to_index(piece_type, is_white)]
    }

    pub fn get_all_pieces(&self) -> Vec<Piece> {
        self.squarelists
            .into_iter()
            .enumerate()
            .flat_map(|(index, squarelist)| {
                squarelist.into_iter_as_piece(get_index_piece(index), get_index_color(index))
            })
            .collect()
    }
}

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

pub const fn get_index_piece(index: usize) -> PieceType {
    assert!(index <= 0b1011);
    PIECE_TYPE_VARIANTS[index % 6]
    //unsafe { transmute::<u8, PieceType>((index % 6) as u8) }
}

pub const fn get_index_color(index: usize) -> bool {
    assert!(index <= 0b1011);
    index < 0b0110
}

impl Default for Board {
    fn default() -> Self {
        Self {
            bit_boards: [0; 12],
            squarelists: [SquareList::new(); 12],
            side_to_move: true,
            half_moves: 0,
            full_moves: 0,
            en_passant_square: None,
            castling_white_kingside: true,
            castling_white_queenside: true,
            castling_black_kingside: true,
            castling_black_queenside: true,
        }
    }
}

const PIECE_TYPE_VARIANTS: [PieceType; 6] = [
    PieceType::King,
    PieceType::Queen,
    PieceType::Rook,
    PieceType::Bishop,
    PieceType::Knight,
    PieceType::Pawn,
];

#[repr(u8)]
#[allow(unused)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub enum PieceType {
    King,
    Queen,
    Rook,
    Bishop,
    Knight,
    Pawn,
}

impl PieceType {
    fn algebraic_name(&self) -> &'static str {
        match self {
            PieceType::King => "K",
            PieceType::Queen => "Q",
            PieceType::Rook => "R",
            PieceType::Bishop => "B",
            PieceType::Knight => "N",
            PieceType::Pawn => "",
        }
    }
}

impl Display for PieceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            PieceType::King => "King",
            PieceType::Queen => "Queen",
            PieceType::Rook => "Rook",
            PieceType::Bishop => "Bishop",
            PieceType::Knight => "Knight",
            PieceType::Pawn => "Pawn",
        };
        write!(f, "{}", name)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord, Hash, Component)]
pub struct Piece {
    square: Square,
    piece_type: PieceType,
    is_white: bool,
}

impl Display for Piece {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut piece_name = self.piece_type().algebraic_name().to_string();
        if !self.is_white() {
            piece_name.make_ascii_lowercase();
        }
        write!(f, "{}{}", piece_name, self.square())
    }
}

impl Piece {
    pub const fn square(&self) -> Square {
        self.square
    }

    pub const fn piece_type(&self) -> PieceType {
        self.piece_type
    }

    pub const fn new(square: Square, piece_type: PieceType, is_white: bool) -> Piece {
        Piece {
            square,
            piece_type,
            is_white,
        }
    }

    pub const fn is_white(&self) -> bool {
        self.is_white
    }

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
}

impl<'a> TryFrom<&'a str> for Board {
    type Error = FENParseError<'a>;

    fn try_from(fen: &'a str) -> Result<Self, Self::Error> {
        let mut board = Board::default();

        let parts: Vec<_> = fen.split_whitespace().collect();
        if parts.len() != 6 {
            return Err(Self::Error::IncorrectAmountOfParts(parts.len()));
        }
        // Note: ranks from 8-1
        let piece_position_part: Vec<_> = parts[0].split('/').collect();
        if piece_position_part.len() != 8 {
            return Err(Self::Error::InvalidAmountOfRanks(piece_position_part.len()));
        }
        for (rank_position, rank) in piece_position_part.into_iter().enumerate() {
            let rank_position = 7 - rank_position;
            let mut file_position = 0;
            for token in rank.chars() {
                let mut increment = 1;
                let piece_type = match token.to_ascii_lowercase() {
                    'k' => Some(PieceType::King),
                    'q' => Some(PieceType::Queen),
                    'r' => Some(PieceType::Rook),
                    'b' => Some(PieceType::Bishop),
                    'n' => Some(PieceType::Knight),
                    'p' => Some(PieceType::Pawn),
                    num @ '1'..='8' => {
                        increment = num
                            .to_digit(10)
                            .expect("Couldn't convert {num} to an int. The match arm should make this impossible.").try_into().unwrap();
                        None
                    }
                    _ => return Err(Self::Error::InvalidRankIToken(token)),
                };
                if let Some(piece_type) = piece_type {
                    #[cfg(test)]
                    println!("chr: {token} | File: {file_position} | Rank: {rank_position} | Is_white: {}", token.is_ascii_uppercase());
                    board.set_square(Piece::new(
                        Square::from_lateral(file_position, rank_position.try_into().unwrap()),
                        piece_type,
                        token.is_ascii_uppercase(),
                    ))
                }
                file_position += increment;
            }
        }

        board.side_to_move = match parts[1]
            .chars()
            .next()
            .expect("2nd part was empty. '.split_whitespace' should make this impossible.")
        {
            'w' => true,
            'b' => false,
            ch => return Err(Self::Error::InvalidCharInSideToMove(ch)),
        };

        // inefficient because the multiple contains check could be formed into one contains check
        // but who cares?
        let castling_ability = parts[2];
        if !(castling_ability == "-"
            || castling_ability
                .chars()
                .all(|ch| ['K', 'Q', 'k', 'q'].contains(&ch)))
        {
            return Err(Self::Error::InvalidCharsInCastlingAbility(castling_ability));
        }

        if castling_ability.len() > 4 {
            return Err(Self::Error::TooManyCharsInCastlingAbility(
                castling_ability.len(),
            ));
        }

        board.castling_white_kingside = castling_ability.contains('K');
        board.castling_white_queenside = castling_ability.contains('Q');
        board.castling_black_kingside = castling_ability.contains('k');
        board.castling_black_queenside = castling_ability.contains('q');

        let en_passant_part = parts[3];
        board.en_passant_square = if en_passant_part == "-" {
            None
        } else {
            Some(Square::try_from_algebraic(en_passant_part)?)
        };

        board.half_moves = parts[4]
            .parse()
            .map_err(Self::Error::HalfMovesIsNotANumber)?;

        board.full_moves = parts[4]
            .parse()
            .map_err(Self::Error::FullMovesIsNotANumber)?;

        Ok(board)
    }
}

#[derive(Clone, Eq, PartialEq, Debug, Error)]
pub enum FENParseError<'a> {
    #[error("{0} parts were given. Expected 6")]
    IncorrectAmountOfParts(usize),
    #[error("Ndddddddddddddddddddd")]
    NoSlash,
    #[error("Letters indicating pieces (k, q, r, b, n, p) or numbers 1-8 were expected, but {0} was given")]
    InvalidRankIToken(char),
    #[error("Expected 8 ranks, but was given {0}")]
    InvalidAmountOfRanks(usize),
    #[error("{0} was given. Expected w or b")]
    InvalidCharInSideToMove(char),
    #[error("'{0}' was given. Expected '-' or up to four of K, Q, k, q (no duplicates)")]
    InvalidCharsInCastlingAbility(&'a str),
    #[error("{0} characters were given. Expected at most 4")]
    TooManyCharsInCastlingAbility(usize),
    #[error("En passant square was not able to be parsed. Reason: {0}")]
    EnPassantSquareParseError(AlgebraicSqaureConversionError),
    #[error("Half moves was not able to be parsed to a u32. {0}")]
    HalfMovesIsNotANumber(ParseIntError),
    #[error("Full moves was not able to be parsed to a u32. {0}")]
    FullMovesIsNotANumber(ParseIntError),
}

impl<'a> From<AlgebraicSqaureConversionError> for FENParseError<'a> {
    fn from(value: AlgebraicSqaureConversionError) -> Self {
        FENParseError::EnPassantSquareParseError(value)
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn create_black_pawn_at_d2() {
        let mut board = Board::default();
        board.set_square(Piece::new(
            Square::from_lateral(3, 1),
            PieceType::Pawn,
            false,
        ));
        let correct_bitboard =
            //   8        7        6        5        4        3        2        1
            //hgfedcba hgfedcba hgfedcba hgfedcba hgfedcba hgfedcba hgfedcba hgfedcba 
            0b00000000_00000000_00000000_00000000_00000000_00000000_00001000_00000000;
        assert_eq!(correct_bitboard, board.get_bitboard(PieceType::Pawn, false))
    }

    #[test]
    fn clear_square_f3() {
        let mut board = Board::default();
        let piece = Piece::new(Square::from_lateral(5, 2), PieceType::Queen, true);
        board.set_square(piece);
        board.clear_square(piece.square());
        assert_eq!(0, board.get_bitboard(piece.piece_type(), piece.is_white()));
    }

    #[test]
    fn set_and_clear_square_of_white_pawn_d2() {
        let mut board = Board::default();
        let piece = Piece::new(Square::from_lateral(3, 1), PieceType::Pawn, true);
        board.set_square(piece);
        board.clear_square_of_piece(piece);
        assert_eq!(0, board.get_bitboard(piece.piece_type(), piece.is_white()));
    }

    #[test]
    fn clear_square_of_black_pawn_f2() {
        let mut board: Board = "8/8/8/8/8/8/pppppppp/8 w - - 0 1"
            .try_into()
            .expect("FEN string should be valid.");
        board.clear_square_of_piece(Piece::new(
            Square::from_lateral(5, 1),
            PieceType::Pawn,
            false,
        ));
        //           hgfedcba
        assert_eq!(0b11011111 << 8, board.get_bitboard(PieceType::Pawn, false));
    }

    #[test]
    fn fen_2_kings_no_castle_white_to_play() {
        let board: Board = "k7/8/8/8/8/8/8/K7 w - - 0 1"
            .try_into()
            .expect("FEN string should be valid");
        let bitboard_king_white = board.get_bitboard(PieceType::King, true);
        assert_eq!(
            bitboard_king_white, 1,
            "left: {:064b}
            right: {:064b}",
            bitboard_king_white, 1
        );
    }

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

    #[test]
    fn piece_at_a1() {
        let board: Board = "8/8/8/8/8/8/8/R7 w - - 0 1".try_into().unwrap();
        assert_eq!(board.get_bitboard(PieceType::Rook, true), 1);
    }
}
