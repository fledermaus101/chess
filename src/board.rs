use std::num::ParseIntError;

use thiserror::Error;

use crate::{
    convert_piece_to_index, get_index_color, get_index_piece,
    piece::{Piece, PieceType},
    r#move::{CastleMove, Move},
    square::{AlgebraicSqaureConversionError, Square},
    squarelist::SquareList,
    PIECE_TYPE_VARIANTS,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Board {
    bitboards: [u64; 12],
    squarelists: [SquareList; PIECE_TYPE_VARIANTS.len() * 2],
    side_to_move: bool,
    half_moves: u8, // should never be larger than 100, as that would be a draw
    // full_moves: u32,
    en_passant_square: Option<Square>,
    castling_rights: [bool; 4], // white kingside, white queenside, black kingside, black queenside
}

#[allow(unused)]
impl Board {
    #[must_use]
    pub const fn can_kingside_castle(&self) -> bool {
        self.castling_rights[!self.side_to_move as usize * 2]
    }

    #[must_use]
    pub const fn can_queenside_castle(&self) -> bool {
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

            let mask = 1 << current_square.square();
            if 0 != self.get_bitboard_of_color(self.side_to_move) & mask {
                break;
            }
            moves.push(current_square);
            if 0 != self.get_bitboard_of_color(!self.side_to_move) & mask {
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
            .map(|to_square| {
                Move::new(
                    start_square,
                    to_square,
                    PieceType::Bishop,
                    self.side_to_move,
                    None,
                    CastleMove::None,
                )
            })
            .collect()
    }

    #[must_use]
    fn rook_moves(&self, start_square: Square) -> Vec<Move> {
        [(0, -1), (1, 0), (0, 1), (-1, 0)]
            .into_iter()
            .flat_map(|(file, rank)| self.calculate_sliding(start_square, file, rank))
            .map(|to_square| {
                Move::new(
                    start_square,
                    to_square,
                    PieceType::Rook,
                    self.side_to_move,
                    None,
                    CastleMove::None,
                )
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
        .filter(|to_square| !self.is_occupied_by_friendly(*to_square))
        .map(|to_square| {
            Move::new(
                start_square,
                to_square,
                PieceType::Knight,
                self.side_to_move,
                None,
                CastleMove::None,
            )
        })
        .collect()
    }

    #[must_use]
    fn pawn_moves(&self, start_square: Square) -> Vec<Move> {
        let mut pawn_moves = Vec::with_capacity(12);
        let mut available_moves = Vec::with_capacity(4);

        let start_rank = if self.side_to_move { 1 } else { 6 };
        // single push
        let square_push_single = start_square.add_rank(self.side_multiplier());
        if 0 == self.get_bitboard_all_pieces() & (1 << square_push_single.square()) {
            available_moves.push(square_push_single);

            // double push pawns
            let square_push_double = start_square.add_rank(2 * self.side_multiplier());
            if start_square.rank() == start_rank
                && 0 == self.get_bitboard_all_pieces() & (1 << square_push_double.square())
            {
                available_moves.push(square_push_double);
            }
        }

        let mask_en_passant = self.get_bitboard_of_color(!self.side_to_move)
            // add the en_passant_square as a valid capture target
            | self.en_passant_square.map_or(0, |sq| 1 << sq.square());
        for offset in [-1, 1] {
            // diagonal capture
            if let Ok(to_square) = start_square.try_add_tuple((offset, self.side_multiplier())) {
                if ((1 << to_square.square()) & mask_en_passant) != 0 {
                    available_moves.push(to_square);
                }
            }
        }

        let promotion_pieces = if start_square.rank() == 7 - start_rank {
            &[
                Some(PieceType::Knight),
                Some(PieceType::Bishop),
                Some(PieceType::Rook),
                Some(PieceType::Queen),
            ][..]
        } else {
            &[None][..]
        };

        for mv in available_moves {
            for &promotion_piece in promotion_pieces {
                pawn_moves.push(Move::new(
                    start_square,
                    mv,
                    PieceType::Pawn,
                    self.side_to_move,
                    promotion_piece,
                    CastleMove::None,
                ));
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
        .filter(|sq| !self.is_occupied_by_friendly(*sq))
        .map(|square_to| {
            Move::new(
                square,
                square_to,
                PieceType::King,
                self.side_to_move,
                None,
                CastleMove::None,
            )
        })
        .collect();

        let bitboard = if self.side_to_move {
            self.get_bitboard_all_pieces()
        } else {
            self.get_bitboard_all_pieces().reverse_bits()
        };

        if self.can_kingside_castle() && 0 == bitboard & 0b0000_0110 {
            king_moves.push(Move::new(
                square,
                square.add_file(2),
                PieceType::King,
                self.side_to_move,
                None,
                CastleMove::KingSide,
            ));
        }
        if self.can_queenside_castle() && 0 == bitboard & 0b0000_0111 {
            king_moves.push(Move::new(
                square,
                square.add_file(-2),
                PieceType::King,
                self.side_to_move,
                None,
                CastleMove::QueenSide,
            ));
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
        .map(|mv| {
            Move::new(
                mv.from(),
                mv.to(),
                PieceType::Queen,
                mv.is_white(),
                mv.promotion_piece(),
                mv.is_castling(),
            )
        })
        .collect()
    }

    #[must_use]
    fn legal_moves(&self) -> Vec<Move> {
        // very crude implementation
        // four (five) scenarios
        // 1. Squares the king can move to are under attack => remove
        // 2. Pieces are absolutely pinned to the king => restrict mobility
        // 3. King is in check => restrict all moves to
        //      1. Single check => only allow (filtered) king moves + blocking with pieces
        //      2. Double check => only allow (filtered) king moves
        // Else: no special handling whatsoever => return legal_moves_pseudo
        // TODO: castling
        let mut pseudo = self.legal_moves_pseudo();
        let copy = Self {
            side_to_move: !self.side_to_move,
            ..*self
        };
        let other_pseudo: Vec<_> = copy
            .legal_moves_pseudo()
            .into_iter()
            .map(|x| x.to())
            .collect();
        // remove moves made by the king to squares attacked by opponent
        pseudo.retain(|x| !(other_pseudo.contains(&x.to()) && x.piece_type() == PieceType::King));

        // remove moves that put our king into check
        pseudo.retain(|x| {
            let mut copy = *self;
            copy.make_move(*x);

            let king_square = copy
                .get_pieces_of_color(!copy.side_to_move)
                .into_iter()
                .find(|piece| piece.piece_type() == PieceType::King)
                .expect("A board should always contain a king")
                .square();

            !copy
                .legal_moves_pseudo()
                .into_iter()
                .any(|m| m.to() == king_square)
        });

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
        //DONE? (testing?): half move clock (captures, castling)
        //TODO: castling rights
        //TODO: castling in general (rook must be moved)
        //TODO: updating en passant + pawn taking the piece

        self.half_moves += 1;
        //TODO: reset half_moves if castling rights are lost
        if (0 != self.get_bitboard_of_color(!self.side_to_move) & (1 << mv.to().square()))
            || mv.piece_type() == PieceType::Pawn
        {
            self.half_moves = 0;
        }

        if mv.piece_type() == PieceType::King {
            if self.castling_rights[usize::from(!self.side_to_move) * 2]
                || self.castling_rights[1 + usize::from(!self.side_to_move) * 2]
            {
                self.half_moves = 0;
            }
            self.castling_rights[usize::from(!self.side_to_move) * 2] = false;
            self.castling_rights[1 + usize::from(!self.side_to_move) * 2] = false;
            match mv.is_castling() {
                CastleMove::KingSide => todo!(),
                CastleMove::QueenSide => todo!(),
                CastleMove::None => (),
            }
        }

        if mv.piece_type() == PieceType::Rook {
            if mv.from().rank_flip() == Square::from_lateral(0, 0)
                && self.castling_rights[1 + usize::from(!self.side_to_move) * 2]
            {
                self.castling_rights[1 + usize::from(!self.side_to_move) * 2] = false;
            }

            if mv.from().rank_flip() == Square::from_lateral(7, 0)
                && self.castling_rights[usize::from(!self.side_to_move) * 2]
            {
                self.castling_rights[usize::from(!self.side_to_move) * 2] = false;
            }
        }

        // en passant
        self.en_passant_square = None;
        if mv.piece_type() == PieceType::Pawn && mv.from().rank().abs_diff(mv.to().rank()) == 2 {
            self.en_passant_square = Some(mv.from().add_rank(self.side_multiplier()));
        }
        self.clear_square(mv.from());
        self.clear_square(mv.to());
        self.set_square(Piece::new(mv.to(), mv.piece_type(), mv.is_white()));
        self.side_to_move = !self.side_to_move;
    }

    #[must_use]
    const fn side_multiplier(&self) -> i8 {
        if self.side_to_move {
            1
        } else {
            -1
        }
    }

    #[must_use]
    const fn is_occupied_by_friendly(&self, square: Square) -> bool {
        0 != self.get_bitboard_of_color(self.side_to_move) & (1 << square.square())
    }

    fn set_square(&mut self, piece: Piece) {
        let index = convert_piece_to_index(piece.piece_type(), piece.is_white());
        self.bitboards[index] |= 1 << piece.square().square();

        self.squarelists[index].add(piece.square());
    }

    fn clear_square_of_piece(&mut self, piece: Piece) {
        let index = convert_piece_to_index(piece.piece_type(), piece.is_white());
        self.bitboards[index] &= !(1 << piece.square().square());

        self.squarelists[index].remove(piece.square());
    }

    fn clear_square(&mut self, square: Square) {
        for bitboard in &mut self.bitboards {
            *bitboard &= !(1 << square.square());
        }
        for squarelist in &mut self.squarelists {
            squarelist.remove(square);
        }
    }

    #[must_use]
    pub const fn get_bitboard(&self, piece_type: PieceType, is_white: bool) -> u64 {
        self.bitboards[convert_piece_to_index(piece_type, is_white)]
    }

    #[must_use]
    pub const fn get_bitboard_of_color(&self, is_white: bool) -> u64 {
        let offset = !is_white as usize * 6;
        let bitboards = self.bitboards;
        // bitboards[offset..6 + offset] // can't use because of const
        //     .iter()
        //     .fold(0, |acc, e| acc | e)
        bitboards[offset]
            | bitboards[1 + offset]
            | bitboards[2 + offset]
            | bitboards[3 + offset]
            | bitboards[4 + offset]
            | bitboards[5 + offset]
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

impl Default for Board {
    fn default() -> Self {
        Self {
            bitboards: [0; 12],
            squarelists: [SquareList::new(); 12],
            side_to_move: true,
            half_moves: 0,
            // full_moves: 0,
            en_passant_square: None,
            castling_rights: [true; 4],
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

        // board.full_moves = fields[4]
        //     .parse()
        //     .map_err(Self::Error::FullMovesIsNotANumber)?;

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

    use super::*;
    use crate::{
        piece::{Piece, PieceType},
        square::Square,
    };

    #[allow(clippy::trivially_copy_pass_by_ref)]
    fn sort_moves(m1: &Move, m2: &Move) -> Ordering {
        m1.to().cmp(&m2.to())
    }

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
        board.set_square(Piece::new(
            Square::from_lateral(7, 5),
            PieceType::Pawn,
            false,
        ));
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
        .map(|(file, rank)| {
            Move::new(
                start_square,
                Square::from_lateral(file, rank),
                PieceType::Bishop,
                true,
                None,
                CastleMove::None,
            )
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
        .map(|(file, rank)| {
            Move::new(
                start_square,
                Square::from_lateral(file, rank),
                PieceType::Rook,
                true,
                None,
                CastleMove::None,
            )
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
        .map(|(file, rank)| {
            Move::new(
                start_square,
                Square::from_lateral(file, rank),
                PieceType::Knight,
                true,
                None,
                CastleMove::None,
            )
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
        board.set_square(Piece::new(
            Square::from_lateral(5, 2),
            PieceType::Knight,
            false,
        ));
        let start_square = Square::from_lateral(4, 1);
        let mut moves = board.pawn_moves(start_square);
        moves.sort_by(sort_moves);

        let mut correct = [(4, 2), (4, 3), (5, 2)].map(|(file, rank)| {
            Move::new(
                start_square,
                Square::from_lateral(file, rank),
                PieceType::Pawn,
                true,
                None,
                CastleMove::None,
            )
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

        let mut correct = [(5, 2), (4, 2)].map(|(file, rank)| {
            Move::new(
                start_square,
                Square::from_lateral(file, rank),
                PieceType::Pawn,
                false,
                None,
                CastleMove::None,
            )
        });
        correct.sort_by(sort_moves);
        assert_eq!(moves, correct);
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
        board.set_square(Piece::new(
            Square::from_lateral(3, 2),
            PieceType::Pawn,
            false,
        ));

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
        .map(|(file, rank)| {
            Move::new(
                start_square,
                Square::from_lateral(file, rank),
                PieceType::Rook,
                true,
                None,
                CastleMove::None,
            )
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
        let expected = [Move::new(
            Square::from_lateral(0, 0),
            Square::from_lateral(0, 1),
            PieceType::King,
            true,
            None,
            CastleMove::None,
        )];

        test_board("1rk5/8/8/8/8/8/8/K7 w - - 0 1", expected.to_vec());
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
        let expected = [
            Move::new(
                Square::from_lateral(0, 0),
                Square::from_lateral(0, 1),
                PieceType::King,
                true,
                None,
                CastleMove::None,
            ),
            Move::new(
                Square::from_lateral(2, 2),
                Square::from_lateral(2, 3),
                PieceType::Pawn,
                true,
                None,
                CastleMove::None,
            ),
        ];
        test_board("1rk5/8/8/8/8/2P5/8/K7 w - - 0 1", expected.to_vec());
    }

    #[test]
    fn legal_moves_bishop_pin_with_checks() {
        // rank
        // |
        // 7 8 0 r k 0 0 0 0 0
        // 6 7 0 | 0 0 0 0 0 0
        // 5 6 0 | 0 0 0 0 0 0
        // 4 5 0 | 0 0 0 0 0 0
        // 3 4 0 | 0 b 0 0 0 0
        // 2 3 0 | P 0 0 0 0 0
        // 1 2 x | 0 0 0 0 0 0
        // 0 1 K | 0 0 0 0 0 0
        //     a b c d e f g h - file
        //     0 1 2 3 4 5 6 7
        let expected = [
            Move::new(
                Square::from_lateral(0, 0),
                Square::from_lateral(0, 1),
                PieceType::King,
                true,
                None,
                CastleMove::None,
            ),
            Move::new(
                Square::from_lateral(2, 2),
                Square::from_lateral(3, 3),
                PieceType::Pawn,
                true,
                None,
                CastleMove::None,
            ),
        ];
        test_board("1rk5/8/8/8/3b4/2P5/8/K7 w - - 0 1", expected.to_vec());
    }

    #[test]
    fn pinned_bishop_with_rook() {
        // to move: black
        // rank
        // |
        // 7 8 0 0 0 0 x k x 0
        // 6 7 0 0 0 0 x b x 0
        // 5 6 0 0 0 0 0 | 0 0
        // 4 5 0 0 0 0 0 | 0 0
        // 3 4 0 0 0 0 0 | 0 0
        // 2 3 0 0 0 0 0 | 0 0
        // 1 2 0 0 0 0 0 | 0 0
        // 0 1 0 0 0 0 K R 0 0
        //     a b c d e f g h - file
        //     0 1 2 3 4 5 6 7
        let expected = [(4, 6), (4, 7), (6, 6), (6, 7)].map(|(file, rank)| {
            Move::new(
                Square::from_lateral(5, 7),
                Square::from_lateral(file, rank),
                PieceType::King,
                false,
                None,
                CastleMove::None,
            )
        });

        test_board("5k2/5b2/8/8/8/8/8/4KR2 b - - 0 1", expected.to_vec());
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
                .seed(u32::from(m.from().square()))
                .luminosity(match m.is_white() {
                    true => Luminosity::Light,
                    false => Luminosity::Dark,
                })
                .to_rgb_array();

            // average the colors column
            // this is wildly unproportional if there are more than 2 colors
            if let Some(other) =
                drawing_board[m.to().square() as usize].and_then(|sq_drawing| sq_drawing.background)
            {
                rgb = [(rgb[0], other[0]), (rgb[1], other[1]), (rgb[2], other[2])]
                    .map(|(a, b)| a / 2 + b / 2);
            }

            let square_drawing = SquareDrawing {
                symbol: 'x',
                background: Some(rgb),
                ..Default::default()
            };

            drawing_board[m.to().square() as usize] = Some(square_drawing);
        }

        for piece in piece_board.get_all_pieces() {
            let mut square_drawing =
                drawing_board[piece.square().square() as usize].unwrap_or_default();

            #[allow(clippy::match_bool)]
            let rgb = RandomColor::new()
                .seed(u32::from(piece.square().square()))
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

            drawing_board[piece.square().square() as usize] = Some(square_drawing);
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

    #[test]
    #[ignore] // makes testing slow
    fn perft_starting_position() {
        let mut board = Board::try_from("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            .expect("Should be valid");
        let correct = [1, 20, 400, 8902, 197_281];
        for (depth, n_nodes) in correct.into_iter().enumerate() {
            assert_eq!(n_nodes, perft(&mut board, depth));
        }
    }

    fn perft(board: &Board, depth: usize) -> u32 {
        if depth == 0 {
            return 1;
        }

        let mut ncount = 0;
        let moves = board.legal_moves();
        print_moves(board, &moves);
        for m in moves {
            let mut board_tmp = *board;
            board_tmp.make_move(m);
            ncount += perft(&mut board_tmp, depth - 1);
        }
        ncount
    }

    #[test]
    fn is_occupied_by_friendly_test() {
        let mut board = Board::default();
        let square = Square::from_lateral(7, 3);
        board.set_square(Piece::new(square, PieceType::Bishop, true));

        assert!(board.is_occupied_by_friendly(square));
    }

    #[test]
    fn legal_moves_under_check() {
        // rank
        // |
        // 7 8 0 0 0 r r k 0 0
        // 6 7 0 0 0 x x 0 0 0
        // 5 6 0 B 0 x x 0 0 0
        // 4 5 0 0 0 x x 0 0 0
        // 3 4 0 0 X X x 0 0 0
        // 2 3 0 0 X K x 0 0 0
        // 1 2 0 0 X 0 x 0 0 0
        // 0 1 0 0 0 0 x 0 0 0
        //     a b c d e f g h - file
        //     0 1 2 3 4 5 6 7
        let expected_king = [(2, 1), (2, 2), (2, 3)].map(|(file, rank)| {
            Move::new(
                Square::from_lateral(3, 2),
                Square::from_lateral(file, rank),
                PieceType::King,
                true,
                None,
                CastleMove::None,
            )
        });

        let expected_bishop = [(3, 3), (3, 7)].map(|(file, rank)| {
            Move::new(
                Square::from_lateral(1, 5),
                Square::from_lateral(file, rank),
                PieceType::Bishop,
                true,
                None,
                CastleMove::None,
            )
        });
        let expected: Vec<Move> = expected_king.into_iter().chain(expected_bishop).collect();

        test_board("3rrk2/8/1B6/8/8/3K4/8/8 w - - 0 1", expected);
    }

    fn test_board(fen: &str, mut expected: Vec<Move>) {
        let board =
            Board::try_from(fen).expect("badly built test. FEN string could not be parsed.");
        let mut actual = board.legal_moves();
        actual.sort_by(sort_moves);
        expected.sort_by(sort_moves);

        println!();
        println!("##### computed board #####");
        print_moves(&board, &actual);
        println!("##### expected board #####");
        print_moves(&board, &expected);
        //println!("{:?}", expected.to_vec().retain(|m| !actual.contains(m)));
        assert_eq!(actual, expected);
    }
}
