#![feature(iter_collect_into)]
#![feature(const_option)]
#![feature(array_chunks)]
pub mod piecelist;

use crate::piecelist::SquareList;
use std::{
    fmt::Debug,
    fmt::Display,
    num::ParseIntError,
    ops::{Add, Sub},
};

use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Move {
    from: Square,
    to: Square,
    piece_type: PieceType,
    is_white: bool,
    promotion_piece: Option<PieceType>,
    is_castling: CastleMove,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CastleMove {
    KingSide,
    QueenSide,
    None,
}

impl Move {
    #[must_use]
    pub const fn piece(&self) -> Piece {
        Piece {
            square: self.to,
            piece_type: self.piece_type,
            is_white: self.is_white,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Board {
    bit_boards: [u64; 12],
    squarelists: [SquareList; PIECE_TYPE_VARIANTS.len() * 2],
    side_to_move: bool,
    half_moves: u8, // should never be larger than 100, as that would be a draw
    full_moves: u32,
    en_passant_square: Option<Square>,
    castling_rights: [bool; 4], // white kingside, white queenside, black kingside, black queenside
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub struct Square(u8);

impl Add for Square {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::try_from_square(self.0 + rhs.0).expect(
            "(Square + Square) was expected to be inside the chess board, but was over the maximum",
        )
    }
}

impl Add<(i8, i8)> for Square {
    type Output = Self;

    fn add(self, rhs: (i8, i8)) -> Self::Output {
        self.try_add_tuple(rhs).expect(
            "(Square + tuple) was expected to be inside the chess board, but was over the maximum",
        )
    }
}

impl Sub for Square {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::try_from_square(self.0 - rhs.0).expect(
            "(Square - Square) was expected to be inside the chess board, but was under the minimum",
        )
    }
}

impl Sub<(i8, i8)> for Square {
    type Output = Self;

    fn sub(self, rhs: (i8, i8)) -> Self::Output {
        let (file, rank) = rhs;
        self.try_add_tuple((-file, -rank))
            .expect("(Square - tuple) was expected to be inside the chess board, but wasn't")
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
    /// Returns a square given a flattened coordinate
    ///
    /// # Panics
    ///
    /// Panics if `square_index > 63`
    #[must_use]
    pub const fn from_square(square_index: u8) -> Self {
        Self::try_from_square(square_index)
            .expect("Square was expected to be inside the chess board, but was over the maximum")
    }

    /// Returns a square given a flattened coordinate if `square_index <= 63`
    #[must_use]
    pub const fn try_from_square(square_index: u8) -> Option<Self> {
        if square_index <= 63 {
            Some(Self(square_index))
        } else {
            None
        }
    }

    /// Returns a square given lateral coordinates
    ///
    /// # Errors
    ///
    /// This function will return an error if `file > 7 || rank > 7`
    pub const fn try_from_lateral(
        file: u8,
        rank: u8,
    ) -> Result<Self, LateralPositionToSquareConversionError> {
        if file > 7 {
            return Err(LateralPositionToSquareConversionError::FileTooLarge(file));
        }
        if rank > 7 {
            return Err(LateralPositionToSquareConversionError::RankTooLarge(rank));
        }
        Ok(Self(rank * 8 + file))
    }

    /// Returns a square given lateral coordinates
    ///
    /// # Panics
    ///
    /// Panics if `file > 7 || rank > 7`
    #[must_use]
    pub const fn from_lateral(file: u8, rank: u8) -> Self {
        assert!(file <= 7, "file cannot be larger than 7 (0 indexed)");
        assert!(rank <= 7, "rank cannot be larger than 7 (0 indexed)");
        Self(rank * 8 + file)
    }

    #[must_use]
    pub const fn file(self) -> u8 {
        self.0 % 8
    }

    #[must_use]
    pub const fn rank(self) -> u8 {
        self.0 / 8
    }

    #[must_use]
    pub const fn to_tuple(self) -> (u8, u8) {
        (self.file(), self.rank())
    }

    /// Returns a new square with the added file offset
    ///
    /// # Panics
    ///
    /// Panics if [`file`](Self) + offset is not inside `0..=7`
    #[must_use]
    pub fn add_file(self, offset: i8) -> Self {
        self.try_add_tuple((offset, 0))
            .expect("Cannot add file offset")
    }

    /// Returns a new square with the added rank offset
    ///
    /// # Panics
    ///
    /// Panics if `[rank](Self) + offset` is not inside `0..=7`
    #[must_use]
    pub fn add_rank(self, offset: i8) -> Self {
        self.try_add_tuple((0, offset))
            .expect("Cannot add rank offset")
    }

    #[must_use]
    pub const fn try_add(self, rhs: Self) -> Option<Self> {
        Self::try_from_square(self.0 + rhs.0)
    }

    /// Adds the square and tuple together
    ///
    /// # Errors
    ///
    /// This function will return an error if the resulting square is out of bounds, meaning one of
    /// these conditions is true
    /// - `file` is not inside `0..=7`
    /// - `rank` is not inside `0..=7`
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
            (Ok(_), Err(err)) | (Err(err), _) => Err(err),
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

    /// Returns a square given a square name
    ///
    /// # Panics
    ///
    /// See [`try_from_algebraic`](Self)
    #[must_use]
    pub fn from_algebraic(square: &str) -> Self {
        Self::try_from_algebraic(square).expect("Square was not able to be parsed.")
    }

    /// Returns a square given a square name
    ///
    /// # Errors
    ///
    /// This function will return an error if
    /// - `square[0]` is not one of `'a'..='h'`
    /// - `square[1]` is not one of `'1'..='8'`
    #[allow(clippy::missing_panics_doc)]
    pub fn try_from_algebraic(square: &str) -> Result<Self, AlgebraicSqaureConversionError> {
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
        Ok(Self::from_lateral(
            file,
            u8::try_from(rank - 1).expect("Above check should make this impossible"),
        ))
    }
}

#[allow(unused)]
impl Board {
    #[must_use]
    pub const fn has_kingside_castle_right(&self) -> bool {
        self.castling_rights[!self.side_to_move as usize * 2]
    }

    #[must_use]
    pub const fn has_queenside_castle_right(&self) -> bool {
        self.castling_rights[1 + !self.side_to_move as usize * 2]
    }

    #[must_use]
    fn calculate_sliding(
        self,
        start_square: Square,
        offset_file: i8,
        offset_rank: i8,
    ) -> Vec<Square> {
        //TODO: collision with friendly and enemy pieces
        // if offset_file and offset_rank are 0, then infinite loop :(
        debug_assert!(!(offset_file == 0 && offset_rank == 0));
        let mut moves = Vec::with_capacity(13);
        let mut current_square = start_square;
        while let Ok(sq) = current_square.try_add_tuple((offset_file, offset_rank)) {
            current_square = sq;

            let mask = 1 << current_square.0;
            if 0 != self.get_bitboard_of_color(self.side_to_move) & mask {
                #[cfg(test)]
                println!("Friendly at {current_square}");
                break;
            }
            moves.push(current_square);
            if 0 != self.get_bitboard_of_color(!self.side_to_move) & mask {
                #[cfg(test)]
                println!("Enemy at {current_square}");
                break;
            }
        }
        moves
    }

    #[must_use]
    fn bishop_moves(&self, start_square: Square) -> Vec<Move> {
        [(-1, -1), (1, -1), (-1, 1), (1, 1)]
            .into_iter()
            .flat_map(|(file, rank)| self.calculate_sliding(start_square, file, rank))
            .map(|to_square| Move {
                from: start_square,
                to: to_square,
                piece_type: PieceType::Bishop,
                is_white: self.side_to_move,
                promotion_piece: None,
                is_castling: CastleMove::None,
            })
            .collect()
    }

    #[must_use]
    fn rook_moves(&self, start_square: Square) -> Vec<Move> {
        [(0, -1), (1, 0), (0, 1), (-1, 0)]
            .into_iter()
            .flat_map(|(file, rank)| self.calculate_sliding(start_square, file, rank))
            .map(|to_square| Move {
                from: start_square,
                to: to_square,
                piece_type: PieceType::Rook,
                is_white: self.side_to_move,
                promotion_piece: None,
                is_castling: CastleMove::None,
            })
            .collect()
    }

    #[must_use]
    fn knight_moves(&self, start_square: Square) -> Vec<Move> {
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
        .filter_map(|offset| start_square.try_add_tuple(offset).ok())
        .map(|to_square| Move {
            from: start_square,
            to: to_square,
            piece_type: PieceType::Knight,
            is_white: self.side_to_move,
            promotion_piece: None,
            is_castling: CastleMove::None,
        })
        .collect()
    }

    #[must_use]
    fn pawn_moves(&self, start_square: Square) -> Vec<Move> {
        let mut pawn_moves = Vec::with_capacity(12);
        // i.e. in which directions can the pawn move?
        let mut available_moves = Vec::with_capacity(4);

        let start_rank = if self.side_to_move { 1 } else { 6 };
        // double push pawns
        if start_square.rank() == start_rank {
            available_moves.push(start_square.add_rank(2 * self.side_multiplier()));
        }

        let mut bitboard = self.get_bitboard_of_color(!self.side_to_move)
            // add the en_passant_square as a valid capture target
            | self.en_passant_square.map_or(0, |sq| 1 << sq.0);
        available_moves.push(start_square.add_rank(self.side_multiplier()));
        for offset in [-1, 1] {
            // diagonal capture
            if let Ok(to_square) = start_square.try_add_tuple((offset, self.side_multiplier())) {
                if (1 << to_square.0 & bitboard) != 0 {
                    available_moves.push(to_square);
                }
            }
        }

        let promotion_pieces = if start_square.rank() == 7 - start_rank {
            [
                Some(PieceType::Knight),
                Some(PieceType::Bishop),
                Some(PieceType::Rook),
                Some(PieceType::Queen),
            ]
            .as_slice()
        } else {
            [None].as_slice()
        };

        for mv in available_moves {
            for &promotion_piece in promotion_pieces {
                pawn_moves.push(Move {
                    from: start_square,
                    to: mv,
                    piece_type: PieceType::Pawn,
                    is_white: self.side_to_move,
                    promotion_piece,
                    is_castling: CastleMove::None,
                });
            }
        }

        pawn_moves
    }

    #[must_use]
    fn king_moves(&self, square: Square) -> Vec<Move> {
        let mut king_moves: Vec<_> = [
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
        .filter_map(|offset| square.try_add_tuple(offset).ok())
        .map(|square_to| Move {
            from: square,
            to: square_to,
            piece_type: PieceType::King,
            is_white: self.side_to_move,
            promotion_piece: None,
            is_castling: CastleMove::None,
        })
        .collect();
        if self.has_kingside_castle_right() {
            king_moves.push(Move {
                from: square,
                to: square.add_file(2),
                piece_type: PieceType::King,
                is_white: self.side_to_move,
                promotion_piece: None,
                is_castling: CastleMove::KingSide,
            });
        }
        if self.has_queenside_castle_right() {
            king_moves.push(Move {
                from: square,
                to: square.add_file(-2),
                piece_type: PieceType::King,
                is_white: self.side_to_move,
                promotion_piece: None,
                is_castling: CastleMove::QueenSide,
            });
        }
        king_moves
    }

    #[must_use]
    fn queen_moves(&self, start_square: Square) -> Vec<Move> {
        [
            self.rook_moves(start_square),
            self.bishop_moves(start_square),
        ]
        .into_iter()
        .flatten()
        .map(|mv| Move {
            piece_type: PieceType::Queen,
            ..mv
        })
        .collect()
    }

    #[must_use]
    fn legal_moves(&self) -> Vec<Move> {
        // very crude implementation
        let mut pseudo = self.legal_moves_pseudo();
        let mut copy = *self;
        copy.side_to_move = !copy.side_to_move;
        let other_pseudo: Vec<_> = copy
            .legal_moves_pseudo()
            .into_iter()
            .map(|x| x.to)
            .collect();
        pseudo.retain(|x| !other_pseudo.contains(&x.to));

        pseudo
    }

    #[must_use]
    fn legal_moves_pseudo(&self) -> Vec<Move> {
        let mut pseudolegal_moves = Vec::new();
        for piece in self.get_pieces_of_color(self.side_to_move) {
            let mut moves: Vec<Move> = match piece.piece_type() {
                PieceType::King => Self::king_moves,
                PieceType::Queen => Self::queen_moves,
                PieceType::Rook => Self::rook_moves,
                PieceType::Bishop => Self::bishop_moves,
                PieceType::Knight => Self::knight_moves,
                PieceType::Pawn => Self::pawn_moves,
            }(self, piece.square());

            pseudolegal_moves.append(&mut moves);
        }
        pseudolegal_moves
    }

    fn make_move(&mut self, mv: Move) {
        self.clear_square(mv.from);
        self.set_square(mv.piece());
    }

    #[must_use]
    const fn side_multiplier(&self) -> i8 {
        if self.side_to_move {
            1
        } else {
            -1
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
        for bit_board in &mut self.bit_boards {
            *bit_board &= !(1 << square.0);
        }
        for squarelist in &mut self.squarelists {
            squarelist.remove(square);
        }
    }

    #[must_use]
    pub const fn get_bitboard(&self, piece_type: PieceType, is_white: bool) -> u64 {
        self.bit_boards[convert_piece_to_index(piece_type, is_white)]
    }

    #[must_use]
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

    #[must_use]
    pub const fn get_bitboard_all_pieces(&self) -> u64 {
        self.get_bitboard_of_color(true) | self.get_bitboard_of_color(false)
    }

    #[must_use]
    pub const fn get_piecelist(&self, piece_type: PieceType, is_white: bool) -> SquareList {
        self.squarelists[convert_piece_to_index(piece_type, is_white)]
    }

    #[must_use]
    pub fn get_all_pieces(&self) -> Vec<Piece> {
        self.squarelists
            .into_iter()
            .enumerate()
            .flat_map(|(index, squarelist)| {
                squarelist.into_iter_as_piece(get_index_piece(index), get_index_color(index))
            })
            .collect()
    }

    #[must_use]
    pub fn get_pieces_of_color(&self, is_white: bool) -> Vec<Piece> {
        let offset = usize::from(!is_white) * 6;
        self.squarelists
            .into_iter()
            .enumerate()
            .skip(offset)
            .take(6)
            .flat_map(|(index, squarelist)| {
                squarelist.into_iter_as_piece(get_index_piece(index), get_index_color(index))
            })
            .collect()
    }
}

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
    //unsafe { transmute::<u8, PieceType>((index % 6) as u8) }
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

impl Default for Board {
    fn default() -> Self {
        Self {
            bit_boards: [0; 12],
            squarelists: [SquareList::new(); 12],
            side_to_move: true,
            half_moves: 0,
            full_moves: 0,
            en_passant_square: None,
            castling_rights: [true; 4],
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

impl<'a> TryFrom<&'a str> for Board {
    type Error = FENParseError<'a>;

    // https://www.chessprogramming.org/Forsyth-Edwards_Notation
    fn try_from(fen: &'a str) -> Result<Self, Self::Error> {
        let mut board = Self::default();

        let fields: Vec<_> = fen.split_whitespace().collect();
        if fields.len() != 6 {
            return Err(Self::Error::IncorrectAmountOfParts(fields.len()));
        }
        let piece_position_field: Vec<_> = fields[0].split('/').collect();
        if piece_position_field.len() != 8 {
            return Err(Self::Error::InvalidAmountOfRanks(
                piece_position_field.len(),
            ));
        }
        // Note: ranks from 8-1 so reverse to get 1-8
        for (rank_position, rank) in piece_position_field.into_iter().rev().enumerate() {
            let rank_position = u8::try_from(rank_position)
                .expect("piece_position_field != 8, guarding if failed?");
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
                        increment = num as u8 - b'0'; // works because num < 10
                        None
                    }
                    _ => return Err(Self::Error::InvalidRankIToken(token)),
                };
                if let Some(piece_type) = piece_type {
                    #[cfg(test)]
                    println!("chr: {token} | File: {file_position} | Rank: {rank_position} | Is_white: {}", token.is_ascii_uppercase());
                    board.set_square(Piece::new(
                        Square::from_lateral(file_position, rank_position),
                        piece_type,
                        token.is_ascii_uppercase(),
                    ));
                }
                file_position += increment;
            }
        }

        board.side_to_move = match fields[1]
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
        let castling_ability = fields[2];
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

        board.castling_rights = ['K', 'Q', 'k', 'q'].map(|ch| castling_ability.contains(ch));

        let en_passant_part = fields[3];
        board.en_passant_square = if en_passant_part == "-" {
            None
        } else {
            Some(
                Square::try_from_algebraic(en_passant_part)
                    .map_err(Self::Error::EnPassantSquareParseError)?,
            )
        };

        board.half_moves = fields[4]
            .parse()
            .map_err(Self::Error::HalfMovesIsNotANumber)?;

        board.full_moves = fields[4]
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

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use colored::Colorize;
    use random_color::{options::Luminosity, RandomColor};

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
        #[allow(clippy::unreadable_literal)]
        let correct_bitboard =
            //   8        7        6        5        4        3        2        1
            //hgfedcba hgfedcba hgfedcba hgfedcba hgfedcba hgfedcba hgfedcba hgfedcba 
            0b00000000_00000000_00000000_00000000_00000000_00000000_00001000_00000000;
        assert_eq!(correct_bitboard, board.get_bitboard(PieceType::Pawn, false));
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
        assert_eq!(0b1101_1111 << 8, board.get_bitboard(PieceType::Pawn, false));
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
        let board: Board = "8/8/8/8/8/8/8/R7 w - - 0 1"
            .try_into()
            .expect("FEN String invalid");
        assert_eq!(board.get_bitboard(PieceType::Rook, true), 1);
    }

    #[test]
    fn bishop_test() {
        // Position: e3
        // rank
        // |
        // 7 8 0 0 0 0 0 0 0 0
        // 6 7 \ 0 0 0 0 0 0 0
        // 5 6 0 \ 0 0 0 0 0 /
        // 4 5 0 0 \ 0 0 0 / 0
        // 3 4 0 0 0 \ 0 / 0 0
        // 2 3 0 0 0 0 x 0 0 0
        // 1 2 0 0 0 / 0 \ 0 0
        // 0 1 0 0 / 0 0 0 \ 0
        //     a b c d e f g h - file
        //     0 1 2 3 4 5 6 7
        let start_square = Square::from_lateral(4, 2);
        let mut board = Board::default();
        board.set_square(Piece {
            square: Square::from_lateral(7, 5),
            piece_type: PieceType::Pawn,
            is_white: false,
        });
        let mut moves = board.bishop_moves(start_square);
        moves.sort_by(sort_moves);

        let mut correct: [Move; 11] = [
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
        .map(|(file, rank)| Move {
            from: start_square,
            to: Square::from_lateral(file, rank),
            piece_type: PieceType::Bishop,
            is_white: true,
            promotion_piece: None,
            is_castling: CastleMove::None,
        });
        correct.sort_by(sort_moves);

        assert_eq!(moves, correct);
    }

    #[test]
    fn rook_test() {
        // Position: e3
        // rank
        // |
        // 7 8 0 0 0 0 | 0 0 0
        // 6 7 0 0 0 0 | 0 0 0
        // 5 6 0 0 0 0 | 0 0 0
        // 4 5 0 0 0 0 | 0 0 0
        // 3 4 0 0 0 0 | 0 0 0
        // 2 3 - - - - x - - -
        // 1 2 0 0 0 0 | 0 0 0
        // 0 1 0 0 0 0 | 0 0 0
        //     a b c d e f g h - file
        //     0 1 2 3 4 5 6 7
        let start_square = Square::from_lateral(4, 2);
        let mut moves = Board::default().rook_moves(start_square);
        moves.sort_by(sort_moves);

        let mut correct: [Move; 14] = [
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
        .map(|(file, rank)| Move {
            from: start_square,
            to: Square::from_lateral(file, rank),
            piece_type: PieceType::Rook,
            is_white: true,
            promotion_piece: None,
            is_castling: CastleMove::None,
        });
        correct.sort_by(sort_moves);

        assert_eq!(moves, correct);
    }

    #[test]
    fn knight_test() {
        // Position: e3
        // rank
        // |
        // 7 8 0 0 0 0 0 0 0 0
        // 6 7 0 0 0 0 0 0 0 0
        // 5 6 0 0 0 0 0 0 0 0
        // 4 5 0 0 0 x 0 x 0 0
        // 3 4 0 0 x 0 0 0 x 0
        // 2 3 0 0 0 0 K 0 0 0
        // 1 2 0 0 x 0 0 0 x 0
        // 0 1 0 0 0 x 0 x 0 0
        //     a b c d e f g h - file
        //     0 1 2 3 4 5 6 7
        let start_square = Square::from_lateral(4, 2);
        let mut moves = Board::default().knight_moves(start_square);
        moves.sort_by(sort_moves);

        let mut correct: [Move; 8] = [
            (5, 0),
            (6, 1),
            (6, 3),
            (5, 4),
            (3, 4),
            (2, 3),
            (2, 1),
            (3, 0),
        ]
        .map(|(file, rank)| Move {
            from: start_square,
            to: Square::from_lateral(file, rank),
            piece_type: PieceType::Knight,
            is_white: true,
            promotion_piece: None,
            is_castling: CastleMove::None,
        });
        correct.sort_by(sort_moves);

        assert_eq!(moves, correct);
    }

    #[test]
    fn pawn_test() {
        // Position: e2
        // rank
        // |
        // 7 8 0 0 0 0 0 0 0 0
        // 6 7 0 0 0 0 0 0 0 0
        // 5 6 0 0 0 0 0 0 0 0
        // 4 5 0 0 0 0 0 0 0 0
        // 3 4 0 0 0 0 x 0 0 0
        // 2 3 0 0 0 0 x k 0 0
        // 1 2 0 0 0 0 P 0 0 0
        // 0 1 0 0 0 0 0 0 0 0
        //     a b c d e f g h - file
        //     0 1 2 3 4 5 6 7
        let mut board = Board::default();
        board.set_square(Piece {
            square: Square::from_lateral(5, 2),
            piece_type: PieceType::Knight,
            is_white: false,
        });
        let start_square = Square::from_lateral(4, 1);
        let mut moves = board.pawn_moves(start_square);
        moves.sort_by(sort_moves);

        let mut correct = [(4, 2), (4, 3), (5, 2)].map(|(file, rank)| Move {
            from: start_square,
            to: Square::from_lateral(file, rank),
            piece_type: PieceType::Pawn,
            is_white: true,
            promotion_piece: None,
            is_castling: CastleMove::None,
        });
        correct.sort_by(sort_moves);

        assert_eq!(moves, correct);
    }

    #[test]
    fn pawn_en_passant_test() {
        // Position: f4
        // rank
        // |
        // 7 8 0 0 0 0 0 0 0 0
        // 6 7 0 0 0 0 0 0 0 0
        // 5 6 0 0 0 0 0 0 0 0
        // 4 5 0 0 0 0 0 0 0 0
        // 3 4 0 0 0 0 P p 0 0
        // 2 3 0 0 0 0 x x 0 0
        // 1 2 0 0 0 0 0 0 0 0
        // 0 1 0 0 0 0 0 0 0 0
        //     a b c d e f g h - file
        //     0 1 2 3 4 5 6 7
        let board: Board = "8/8/8/8/4Pp2/8/8/8 b - e3 0 1"
            .try_into()
            .expect("FEN String invalid");

        let start_square = Square::from_lateral(5, 3);
        let mut moves = board.pawn_moves(start_square);
        moves.sort_by(sort_moves);

        let mut correct = [(5, 2), (4, 2)].map(|(file, rank)| Move {
            from: start_square,
            to: Square::from_lateral(file, rank),
            piece_type: PieceType::Pawn,
            is_white: false,
            promotion_piece: None,
            is_castling: CastleMove::None,
        });
        correct.sort_by(sort_moves);
        assert_eq!(moves, correct);
    }

    #[allow(clippy::trivially_copy_pass_by_ref)]
    fn sort_moves(m1: &Move, m2: &Move) -> Ordering {
        m1.to.cmp(&m2.to)
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

    #[test]
    fn rook_movement_with_enemies_as_blocker() {
        // Position: d5
        // rank
        // |
        // 7 8 0 0 0 | 0 0 0 0
        // 6 7 0 0 0 | 0 0 0 0
        // 5 6 - - - x - - - -
        // 4 5 0 0 0 | 0 0 0 0
        // 3 4 0 0 0 | 0 0 0 0
        // 2 3 0 0 0 B 0 0 0 0
        // 1 2 0 0 0 0 0 0 0 0
        // 0 1 0 0 0 0 0 0 0 0
        //     a b c d e f g h - file
        //     0 1 2 3 4 5 6 7
        let start_square = Square::from_lateral(3, 5);
        let mut board = Board::default();
        board.set_square(Piece {
            square: Square::from_lateral(3, 2),
            piece_type: PieceType::Pawn,
            is_white: false,
        });

        let mut actual: Vec<_> = board.rook_moves(start_square);
        actual.sort_by(sort_moves);

        let mut expected = [
            (0, 5),
            (1, 5),
            (2, 5),
            (4, 5),
            (5, 5),
            (6, 5),
            (7, 5),
            (3, 7),
            (3, 6),
            (3, 4),
            (3, 3),
            (3, 2),
        ]
        .map(|(file, rank)| Move {
            from: start_square,
            to: Square::from_lateral(file, rank),
            piece_type: PieceType::Rook,
            is_white: true,
            promotion_piece: None,
            is_castling: CastleMove::None,
        });
        expected.sort_by(sort_moves);

        assert_eq!(actual, expected);
    }

    #[test]
    fn legal_moves_kings_with_checks() {
        // rank
        // |
        // 7 8 - r k 0 0 0 0 0
        // 6 7 0 | 0 0 0 0 0 0
        // 5 6 0 | 0 0 0 0 0 0
        // 4 5 0 | 0 0 0 0 0 0
        // 3 4 0 | 0 0 0 0 0 0
        // 2 3 0 | 0 0 0 0 0 0
        // 1 2 0 | 0 0 0 0 0 0
        // 0 1 K | 0 0 0 0 0 0
        //     a b c d e f g h - file
        //     0 1 2 3 4 5 6 7
        let board = Board::try_from("1rk5/8/8/8/8/8/8/K7 w - - 0 1").expect("Should be valid");

        let mut expected = [Move {
            from: Square::from_lateral(0, 0),
            to: Square::from_lateral(0, 1),
            piece_type: PieceType::King,
            is_white: true,
            promotion_piece: None,
            is_castling: CastleMove::None,
        }];
        expected.sort_by(sort_moves);

        let mut actual = board.legal_moves();
        actual.sort_by(sort_moves);

        assert_eq!(actual, expected);
    }

    #[test]
    fn legal_moves_kings_and_pawn_with_checks() {
        // rank
        // |
        // 7 8 0 r k 0 0 0 0 0
        // 6 7 0 | 0 0 0 0 0 0
        // 5 6 0 | 0 0 0 0 0 0
        // 4 5 0 | 0 0 0 0 0 0
        // 3 4 0 | x 0 0 0 0 0
        // 2 3 0 | P 0 0 0 0 0
        // 1 2 x | 0 0 0 0 0 0
        // 0 1 K | 0 0 0 0 0 0
        //     a b c d e f g h - file
        //     0 1 2 3 4 5 6 7
        let board = Board::try_from("1rk5/8/8/8/8/2P5/8/K7 w - - 0 1").expect("Should be valid");
        let mut expected = [
            Move {
                from: Square::from_lateral(0, 0),
                to: Square::from_lateral(0, 1),
                piece_type: PieceType::King,
                is_white: true,
                promotion_piece: None,
                is_castling: CastleMove::None,
            },
            Move {
                from: Square::from_lateral(2, 2),
                to: Square::from_lateral(2, 3),
                piece_type: PieceType::Pawn,
                is_white: true,
                promotion_piece: None,
                is_castling: CastleMove::None,
            },
        ];
        expected.sort_by(sort_moves);

        let mut actual = board.legal_moves();
        actual.sort_by(sort_moves);

        print_moves(&board, &actual);
        print_moves(&board, &expected);
        assert_eq!(actual, expected);
    }

    #[test]
    fn legal_moves_bishop_pin_with_checks() {
        // rank
        // |
        // 7 8 0 r k 0 0 0 0 0
        // 6 7 0 | 0 0 0 0 0 0
        // 5 6 0 | 0 0 0 0 0 0
        // 4 5 0 | 0 0 0 0 0 0
        // 3 4 0 | x b 0 0 0 0
        // 2 3 0 | P 0 0 0 0 0
        // 1 2 0 | 0 0 0 0 0 0
        // 0 1 K | 0 0 0 0 0 0
        //     a b c d e f g h - file
        //     0 1 2 3 4 5 6 7
        let board = Board::try_from("1rk5/8/8/8/3b4/2P5/8/K7 w - - 0 1").expect("Should be valid");
        let mut expected = [
            Move {
                from: Square::from_lateral(0, 0),
                to: Square::from_lateral(0, 1),
                piece_type: PieceType::King,
                is_white: true,
                promotion_piece: None,
                is_castling: CastleMove::None,
            },
            Move {
                from: Square::from_lateral(2, 2),
                to: Square::from_lateral(2, 3),
                piece_type: PieceType::Pawn,
                is_white: true,
                promotion_piece: None,
                is_castling: CastleMove::None,
            },
            Move {
                from: Square::from_lateral(2, 2),
                to: Square::from_lateral(3, 3),
                piece_type: PieceType::Pawn,
                is_white: true,
                promotion_piece: None,
                is_castling: CastleMove::None,
            },
        ];
        expected.sort_by(sort_moves);

        let mut actual = board.legal_moves();
        actual.sort_by(sort_moves);

        print_moves(&board, &actual);
        print_moves(&board, &expected);
        assert_eq!(actual, expected);
    }

    #[allow(dead_code)]
    fn print_moves(piece_board: &Board, moves: &[Move]) {
        // This could be done, but it would need cleverness beyond the scope of a debugging
        // utility.
        // #[derive(Debug)]
        // enum Direction {
        //     Horizontal,     // -
        //     Vertical,       // |
        //     DiagonalLeft,   // ⟍
        //     DiagonalRight,  // ⟋
        // }                   // ⤫
        //                     // +
        #[derive(Clone, Copy)]
        struct SquareDrawing {
            symbol: char,
            foreground: [u8; 3],
            background: Option<[u8; 3]>,
        }

        impl Default for SquareDrawing {
            fn default() -> Self {
                Self {
                    symbol: '0',
                    foreground: [80, 80, 80],
                    background: None,
                }
            }
        }

        let mut drawing_board: [Option<SquareDrawing>; 64] = [None; 64];

        for m in moves {
            #[allow(clippy::match_bool)]
            let mut rgb = RandomColor::new()
                .seed(u32::from(m.from.0))
                .luminosity(match m.is_white {
                    true => Luminosity::Light,
                    false => Luminosity::Dark,
                })
                .to_rgb_array();

            // average the colors column
            // this is wildly unproportional if there are more than 2 colors
            if let Some(other) =
                drawing_board[m.to.0 as usize].and_then(|sq_drawing| sq_drawing.background)
            {
                rgb = [(rgb[0], other[0]), (rgb[1], other[1]), (rgb[2], other[2])]
                    .map(|(a, b)| a / 2 + b / 2);
            }

            let square_drawing = SquareDrawing {
                symbol: 'x',
                background: Some(rgb),
                ..Default::default()
            };

            drawing_board[m.to.0 as usize] = Some(square_drawing);
        }

        for piece in piece_board.get_all_pieces() {
            let mut square_drawing = drawing_board[piece.square.0 as usize].unwrap_or_default();

            #[allow(clippy::match_bool)]
            let rgb = RandomColor::new()
                .seed(u32::from(piece.square().0))
                .luminosity(match piece.is_white() {
                    true => Luminosity::Light,
                    false => Luminosity::Dark,
                })
                .to_rgb_array();

            square_drawing = SquareDrawing {
                foreground: rgb,
                symbol: piece.piece_letter(),
                ..square_drawing
            };

            drawing_board[piece.square.0 as usize] = Some(square_drawing);
        }
        let drawing_board = drawing_board.map(Option::unwrap_or_default);

        // rank
        // |
        // 7 8 - r k 0 0 0 0 0
        // 6 7 0 | 0 0 0 0 0 0
        // 5 6 0 | 0 0 0 0 0 0
        // 4 5 0 | 0 0 0 0 0 0
        // 3 4 0 | 0 b 0 0 0 0
        // 2 3 0 | P 0 0 0 0 0
        // 1 2 0 | 0 0 0 0 0 0
        // 0 1 K | 0 0 0 0 0 0
        //     a b c d e f g h - file
        //     0 1 2 3 4 5 6 7
        println!("rank");
        println!("|");
        for (rank_index, row) in drawing_board.array_chunks::<8>().enumerate().rev() {
            print!("{} {}", rank_index, rank_index + 1);

            for square_drawing in row {
                let [r, g, b] = square_drawing.foreground;
                let mut colored_string = square_drawing.symbol.to_string().truecolor(r, g, b);

                if let Some(background) = square_drawing.background {
                    let [r, g, b] = background;
                    colored_string = colored_string.on_truecolor(r, g, b);
                }
                print!(" {colored_string}");
            }

            println!();
        }
        println!("    a b c d e f g h");
        println!("    0 1 2 3 4 5 6 7 - file");
    }
}
