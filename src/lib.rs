#![feature(iter_collect_into)]
#![feature(const_option)]
pub mod piecelist;

use crate::piecelist::SquareList;
use std::{
    fmt::Display,
    num::ParseIntError,
    ops::{Add, Sub},
};

use bevy::prelude::{Component, Resource};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Move {
    from: Square,
    to: Square,
    piece_type: PieceType,
    is_white: bool,
}

impl Move {
    pub fn piece(&self) -> Piece {
        Piece {
            square: self.to,
            piece_type: self.piece_type,
            is_white: self.is_white,
        }
    }
}

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

impl Add for Square {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::try_from_square(self.0 + rhs.0).expect(
            "Square + Square was expected to be inside the chess board, but was over the maximum",
        )
    }
}

impl Add<(i8, i8)> for Square {
    type Output = Self;

    fn add(self, rhs: (i8, i8)) -> Self::Output {
        self.try_add_tuple(rhs).expect(
            "Square + tuple was expected to be inside the chess board, but was over the maximum",
        )
    }
}

impl Sub for Square {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::try_from_square(self.0 - rhs.0).expect(
            "Square - Square was expected to be inside the chess board, but was under the minimum",
        )
    }
}

impl Sub<(i8, i8)> for Square {
    type Output = Self;

    fn sub(self, rhs: (i8, i8)) -> Self::Output {
        let (file, rank) = rhs;
        self.try_add_tuple(rhs)
            .expect("Square - tuple was expected to be inside the chess board, but wasn't")
    }
}

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

#[derive(Clone, Eq, PartialEq, Debug, Error)]
pub enum LateralPositionToSquareConversionError {
    #[error("file (0 indexed) should not be larger than 7. Actual {0}")]
    FileTooLarge(u8),
    #[error("rank (0 indexed) should not be larger than 7. Actual {0}")]
    RankTooLarge(u8),
    #[error("file should not be negative. Actual {0}")]
    FileNegative(i8),
    #[error("rank should not be negative. Actual {0}")]
    RankNegative(i8),
}

impl Display for Square {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", (self.file() + b'a') as char, self.rank() + 1)
    }
}

impl Square {
    pub fn from_square(square_index: u8) -> Square {
        Self::try_from_square(square_index)
            .expect("Square was expected to be inside the chess board, but was over the maximum")
    }

    pub const fn try_from_square(square_index: u8) -> Option<Square> {
        match square_index <= 63 {
            true => Some(Square(square_index)),
            false => None,
        }
    }

    pub const fn try_from_lateral(
        file: u8,
        rank: u8,
    ) -> Result<Square, LateralPositionToSquareConversionError> {
        if file > 7 {
            return Err(LateralPositionToSquareConversionError::FileTooLarge(file));
        }
        if rank > 7 {
            return Err(LateralPositionToSquareConversionError::RankTooLarge(rank));
        }
        Ok(Square(rank * 8 + file))
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

    pub const fn to_tuple(self) -> (u8, u8) {
        (self.file(), self.rank()) // Don't worry, this gets optimized and isn't wasteful
    }

    pub fn add_file(self, offset: i8) -> Self {
        self.try_add_tuple((offset, 0))
            .expect("Cannot add file offset")
    }

    pub fn add_rank(self, offset: i8) -> Self {
        self.try_add_tuple((0, offset))
            .expect("Cannot add rank offset")
    }

    pub const fn try_add(self, rhs: Self) -> Option<Self> {
        Square::try_from_square(self.0 + rhs.0)
    }

    pub const fn try_add_tuple(
        self,
        rhs: (i8, i8),
    ) -> Result<Self, LateralPositionToSquareConversionError> {
        let (lhs_file, lhs_rank) = self.to_tuple();
        let (rhs_file, rhs_rank) = rhs;
        let file = match lhs_file.checked_add_signed(rhs_file) {
            Some(f) => Ok(f),
            None => Err(LateralPositionToSquareConversionError::FileNegative(
                rhs_file.saturating_add_unsigned(lhs_file),
            )),
        };
        let rank = match lhs_rank.checked_add_signed(rhs_rank) {
            Some(r) => Ok(r),
            None => Err(LateralPositionToSquareConversionError::RankNegative(
                rhs_rank.saturating_add_unsigned(lhs_rank),
            )),
        };
        match (file, rank) {
            (Ok(file), Ok(rank)) => Self::try_from_lateral(file, rank),
            (Ok(_), Err(err)) => Err(err),
            (Err(err), _) => Err(err),
        }
    }

    // pub fn try_add_tuple_non_const(
    //     self,
    //     rhs: (i8, i8),
    // ) -> Result<Self, LateralPositionToSquareConversionError> {
    //     let (lhs_file, lhs_rank) = self.to_tuple();
    //     let (rhs_file, rhs_rank) = rhs;

    //     let file = lhs_file.checked_add_signed(rhs_file).ok_or_else(|| {
    //         LateralPositionToSquareConversionError::FileNegative(
    //             rhs_file.saturating_add_unsigned(lhs_file),
    //         )
    //     })?;
    //     let rank = lhs_rank.checked_add_signed(rhs_rank).ok_or_else(|| {
    //         LateralPositionToSquareConversionError::RankNegative(
    //             rhs_rank.saturating_add_unsigned(lhs_rank),
    //         )
    //     })?;

    //     Self::try_from_lateral(file, rank)
    // }

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
    fn calculate_sliding(square: Square, offset_file: i8, offset_rank: i8) -> Vec<Square> {
        let mut moves = Vec::new();
        let (mut file, mut rank) = square.to_tuple();
        loop {
            match file.checked_add_signed(offset_file) {
                Some(f) if f <= 7 => file = f,
                _ => break,
            };
            match rank.checked_add_signed(offset_rank) {
                Some(r) if r <= 7 => rank = r,
                _ => break,
            }
            moves.push(Square::from_lateral(file, rank));
        }
        moves
    }

    fn bishop_moves(&self, square: Square) -> Vec<Square> {
        let mut moves = Vec::new();

        moves.append(&mut Self::calculate_sliding(square, -1, -1)); // Down left
        moves.append(&mut Self::calculate_sliding(square, 1, -1)); // Up left
        moves.append(&mut Self::calculate_sliding(square, -1, 1)); // Down right
        moves.append(&mut Self::calculate_sliding(square, 1, 1)); // Up right

        moves
    }

    fn rook_moves(&self, square: Square) -> Vec<Square> {
        let mut moves = Vec::new();

        moves.append(&mut Self::calculate_sliding(square, 0, -1)); // Left
        moves.append(&mut Self::calculate_sliding(square, 1, 0)); // Up
        moves.append(&mut Self::calculate_sliding(square, 0, 1)); // Right
        moves.append(&mut Self::calculate_sliding(square, -1, 0)); // Down

        moves
    }

    fn knight_moves(&self, square: Square) -> Vec<Square> {
        [
            (1, -2),
            (2, -1),
            (2, 1),
            (1, 2),
            (-1, 2),
            (-2, 1),
            (-2, -1),
            (-1, -2),
        ]
        .into_iter()
        .filter_map(|offset| square.try_add_tuple(offset).ok())
        .collect()
    }

    fn pawn_moves(&self, square: Square) -> Vec<Square> {
        let mut pawn_moves = Vec::with_capacity(5);
        if [1, 6].contains(&square.rank()) {
            pawn_moves.push(square.add_file(2 * self.side_multiplier()));
        }
        for offset in [-1, 1] {
            let square = square.add_rank(offset);
        }
        pawn_moves.push(square.add_file(1));
        pawn_moves
    }

    fn get_legalmoves(&self) -> Vec<Move> {
        for piece in self.get_all_pieces() {
            let moves: Vec<Square> = match piece.piece_type() {
                PieceType::King => [
                    (-1, -1),
                    (-1, 0),
                    (-1, 1),
                    (0, -1),
                    (0, 1),
                    (1, -1),
                    (1, 0),
                    (1, 1),
                ]
                .into_iter()
                .map(|offset| piece.square() + offset)
                .collect(),
                PieceType::Queen => {
                    let mut rook_moves = self.rook_moves(piece.square());
                    rook_moves.append(&mut self.bishop_moves(piece.square()));
                    rook_moves
                }
                PieceType::Rook => self.rook_moves(piece.square()),
                PieceType::Bishop => self.bishop_moves(piece.square()),
                PieceType::Knight => self.knight_moves(piece.square()),
                PieceType::Pawn => self.pawn_moves(piece.square()),
            };
        }
        todo!()
    }

    fn make_move(&mut self, mv: Move) {
        self.clear_square(mv.from);
        self.set_square(mv.piece())
    }

    fn side_multiplier(&self) -> i8 {
        match self.side_to_move {
            true => 1,
            false => -1,
        }
    }

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

    pub const fn get_bitboard_of_color(&self, is_white: bool) -> u64 {
        let offset = !is_white as usize * 6;
        let bit_boards = self.bit_boards;
        // bit_boards[offset..6 + offset] // can't use because of const
        //     .iter()
        //     .fold(0, |acc, e| acc | e)
        bit_boards[offset]
            | bit_boards[1 + offset]
            | bit_boards[2 + offset]
            | bit_boards[3 + offset]
            | bit_boards[4 + offset]
            | bit_boards[5 + offset]
    }

    pub const fn get_bitboard_all_pieces(&self) -> u64 {
        self.get_bitboard_of_color(true) | self.get_bitboard_of_color(false)
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

    #[test]
    fn bishop_test() {
        // Position: e3
        // 7 0 0 0 0 0 0 0 0
        // 6 \ 0 0 0 0 0 0 0
        // 5 0 \ 0 0 0 0 0 /
        // 4 0 0 \ 0 0 0 / 0
        // 3 0 0 0 \ 0 / 0 0
        // 2 0 0 0 0 x 0 0 0
        // 1 0 0 0 / 0 \ 0 0
        // 0 0 0 / 0 0 0 \ 0
        //   a b c d e f g h
        //   0 1 2 3 4 5 6 7
        let mut moves = Board::default().bishop_moves(Square::from_lateral(4, 2));
        moves.sort();

        let mut correct: [Square; 11] = [
            // down right
            (6, 0),
            (5, 1),
            // top left
            (3, 3),
            (2, 4),
            (1, 5),
            (0, 6),
            // down left
            (3, 1),
            (2, 0),
            // right up
            (5, 3),
            (6, 4),
            (7, 5),
        ]
        .map(|(file, rank)| Square::from_lateral(file, rank));
        correct.sort();

        assert_eq!(moves, correct);
    }

    #[test]
    fn rook_test() {
        // Position: e3
        // 7 0 0 0 0 | 0 0 0
        // 6 0 0 0 0 | 0 0 0
        // 5 0 0 0 0 | 0 0 0
        // 4 0 0 0 0 | 0 0 0
        // 3 0 0 0 0 | 0 0 0
        // 2 - - - - x - - -
        // 1 0 0 0 0 | 0 0 0
        // 0 0 0 0 0 | 0 0 0
        //   a b c d e f g h
        //   0 1 2 3 4 5 6 7
        let mut moves = Board::default().rook_moves(Square::from_lateral(4, 2));
        moves.sort();

        let mut correct: [Square; 14] = [
            // down
            (4, 0),
            (4, 1),
            // up
            (4, 3),
            (4, 4),
            (4, 5),
            (4, 6),
            (4, 7),
            // left
            (0, 2),
            (1, 2),
            (2, 2),
            (3, 2),
            // right
            (5, 2),
            (6, 2),
            (7, 2),
        ]
        .map(|(file, rank)| Square::from_lateral(file, rank));
        correct.sort();

        assert_eq!(moves, correct);
    }

    #[test]
    fn knight_test() {
        // Position: e3
        // 7 0 0 0 0 0 0 0 0
        // 6 0 0 0 0 0 0 0 0
        // 5 0 0 0 0 0 0 0 0
        // 4 0 0 0 x 0 x 0 0
        // 3 0 0 x 0 0 0 x 0
        // 2 0 0 0 0 k 0 0 0
        // 1 0 0 x 0 0 0 x 0
        // 0 0 0 0 x 0 x 0 0
        //   a b c d e f g h
        //   0 1 2 3 4 5 6 7
        let mut moves = Board::default().knight_moves(Square::from_lateral(4, 2));
        moves.sort();

        let mut correct: [Square; 8] = [
            (5, 0),
            (6, 1),
            (6, 3),
            (5, 4),
            (3, 4),
            (2, 3),
            (2, 1),
            (3, 0),
        ]
        .map(|(file, rank)| Square::from_lateral(file, rank));
        correct.sort();

        assert_eq!(moves, correct);
    }

    #[test]
    fn square_add_tuple() {
        assert_eq!(
            Square::from_lateral(2, 3) + (5, 4),
            Square::from_lateral(7, 7)
        );
    }

    #[test]
    fn square_add_tuple_negative() {
        assert_eq!(
            Square::from_lateral(2, 2).try_add_tuple((-3, -1)),
            Err(LateralPositionToSquareConversionError::FileNegative(-1))
        );
        assert_eq!(
            Square::from_lateral(2, 2).try_add_tuple((-1, -3)),
            Err(LateralPositionToSquareConversionError::RankNegative(-1))
        );
        assert_eq!(
            Square::from_lateral(6, 3) + (-6, -2),
            Square::from_lateral(0, 1)
        );
    }

    #[test]
    fn square_add_square() {
        assert_eq!(
            Square::from_lateral(3, 6) + Square::from_lateral(4, 1),
            Square::from_lateral(7, 7)
        );
    }

    #[test]
    fn square_add_square_over_max() {
        assert_eq!(
            Square::from_lateral(4, 6).try_add(Square::from_lateral(4, 1)),
            None
        );
    }
}
