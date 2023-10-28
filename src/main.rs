use bevy::{prelude::*, window::PrimaryWindow};
use chess::*;

const SQUARE_SIZE: f32 = 80.0;
const BOARD_OFFSET: f32 = -450.0;
const PIECE_SIZE: f32 = 333.0;
const START_POSITION: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const PIECE_SCALE: f32 = 0.25;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(ImagePlugin::default_nearest()))
        .insert_resource::<Board>(
            START_POSITION
                .try_into()
                .expect("Invalid hard coded start position"),
        )
        .add_systems(Startup, create_board)
        .add_systems(Startup, load_pieces)
        .add_systems(Update, on_left_press)
        .add_systems(Update, on_left_release)
        .add_systems(Update, hover)
        .add_systems(Update, drag)
        .add_systems(Update, on_drag)
        // .add_systems(Update, on_drag_release)
        .run();
}

#[derive(Component)]
struct BoardTile(Square);

fn create_board(mut commands: Commands) {
    commands.spawn(Camera2dBundle::default());
    for rank in 0..=7 {
        for file in 0..=7 {
            commands.spawn((
                SpriteBundle {
                    sprite: Sprite {
                        color: Color::Rgba {
                            red: if (file + rank) % 2 == 0 { 0.0 } else { 0.39 },
                            green: 0.0,
                            blue: 0.45,
                            alpha: 1.0,
                        },
                        custom_size: Some(Vec2::new(SQUARE_SIZE, SQUARE_SIZE)),
                        ..default()
                    },
                    transform: Transform::from_translation(Vec3::new(
                        BOARD_OFFSET + (file as f32 * SQUARE_SIZE),
                        BOARD_OFFSET + (rank as f32 * SQUARE_SIZE),
                        0.0,
                    )),
                    ..default()
                },
                BoardTile(Square::from_lateral(file, rank)),
            ));
        }
    }
}

fn load_pieces(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut texture_atlases: ResMut<Assets<TextureAtlas>>,
    board: Res<Board>,
) {
    let atlas = TextureAtlas::from_grid(
        asset_server.load("Pieces.png"),
        Vec2::new(PIECE_SIZE, PIECE_SIZE),
        6,
        2,
        None,
        None,
    );
    let atlas_handle = texture_atlases.add(atlas);
    for piece in board.get_all_pieces() {
        commands.spawn(create_piece_bundle(piece, atlas_handle.clone()));
    }

    commands.insert_resource(PieceAtlasHandle(atlas_handle));
}

#[derive(Resource)]
struct PieceAtlasHandle(Handle<TextureAtlas>);

fn create_piece_bundle(
    piece: Piece,
    atlas_handle: Handle<TextureAtlas>,
) -> (Piece, SpriteSheetBundle) {
    const MAP_BETWEEN_PIECE_TYPE_VARIANTS_AND_PIECE_POSITIONS_IN_ATLAS: [usize; 6] =
        [0, 1, 4, 2, 3, 5];
    (
        piece,
        SpriteSheetBundle {
            texture_atlas: atlas_handle,
            transform: Transform {
                translation: Vec3::new(
                    BOARD_OFFSET + (piece.square().file() as f32 * SQUARE_SIZE),
                    BOARD_OFFSET + (piece.square().rank() as f32 * SQUARE_SIZE),
                    1.0,
                ),
                scale: Vec3::new(PIECE_SCALE, PIECE_SCALE, PIECE_SCALE),
                ..default()
            },
            sprite: TextureAtlasSprite::new(
                MAP_BETWEEN_PIECE_TYPE_VARIANTS_AND_PIECE_POSITIONS_IN_ATLAS
                    [piece.piece_type() as usize]
                    + !piece.is_white() as usize * 6,
            ),
            ..default()
        },
    )
}

#[derive(Component)]
struct Dragged;

#[derive(Component)]
struct Hovered;

#[allow(unused)]
fn hover(
    mut commands: Commands,
    q_piece: Query<(Entity, &Transform, &Piece), Without<Dragged>>,
    q_windows: Query<&Window, With<PrimaryWindow>>,
) {
    let q_windows = q_windows.single();
    if let Some(cursor_position) = q_windows.cursor_position() {
        for (entity, transform, piece) in &q_piece {
            let size = PIECE_SIZE * PIECE_SCALE;
            let half_length = size / 2.0;

            let cursor_x = cursor_position.x - q_windows.width() / 2.0;
            let cursor_y = q_windows.height() / 2.0 - cursor_position.y;

            if transform.translation.x - half_length < cursor_x
                && transform.translation.x + half_length > cursor_x
                && transform.translation.y - half_length < cursor_y
                && transform.translation.y + half_length > cursor_y
            {
                // dbg!(transform.translation.truncate());
                commands.entity(entity).insert(Hovered);
            } else {
                commands.entity(entity).remove::<Hovered>();
            }
        }
    }
}

fn on_left_press(
    mut commands: Commands,
    buttons: Res<Input<MouseButton>>,
    q_hover_pieces: Query<(Entity, &Piece), With<Hovered>>,
) {
    if buttons.just_pressed(MouseButton::Left) {
        if let Some((entity, piece)) = q_hover_pieces.iter().next() {
            commands.entity(entity).insert(Dragged);
            println!("Drag   {}", piece);
        }
    }
}

fn on_left_release(
    mut commands: Commands,
    buttons: Res<Input<MouseButton>>,
    mut q_dragged_pieces: Query<(Entity, &Piece), With<Dragged>>,
    mut q_board_tiles: Query<(&mut Sprite, &BoardTile)>,
) {
    if buttons.just_released(MouseButton::Left) {
        if let Some((entity, mut piece)) = q_dragged_pieces.iter_mut().next() {
            commands.entity(entity).remove::<Dragged>();
            println!("Undrag {}", piece);
            {
                // Reset hightlighted square
                let (mut board_sprite, _) = q_board_tiles
                    .iter_mut()
                    .find(|(_, board_tile)| board_tile.0 == piece.square())
                    .expect("Every square should exist.");
                board_sprite.color.set_b(0.45);
            }
            {
                // Set position of the piece
                let (_, board_tile) = q_board_tiles
                    .iter_mut()
                    .find(|(_, board_tile)| {
                        board_tile.0 == todo!("Find the square where the mouse is currently on")
                    })
                    .expect("Every square should exist.");

                let p = &Piece::new(board_tile.0, piece.piece_type(), piece.is_white());
                let _ = std::mem::replace(&mut piece, p);
            }
        }
    }
}

fn cursor_position(cursor_position: Vec2, window: &Window) -> Vec2 {
    Vec2 {
        x: cursor_position.x - window.width() / 2.0,
        y: window.height() / 2.0 - cursor_position.y,
    }
}

fn drag(
    mut q_dragged_pieces: Query<&mut Transform, With<Dragged>>,
    q_windows: Query<&Window, With<PrimaryWindow>>,
) {
    let q_windows = q_windows.single();
    if let Some(mut transform) = q_dragged_pieces.iter_mut().next() {
        if let Some(cursor_position) = q_windows.cursor_position() {
            transform.translation = Vec3::new(
                cursor_position.x - q_windows.width() / 2.0,
                q_windows.height() / 2.0 - cursor_position.y,
                1.0,
            );
        }
    }
}

fn on_drag(
    q_dragged_pieces: Query<&Piece, Added<Dragged>>,
    mut q_board_tiles: Query<(&mut Sprite, &BoardTile)>,
) {
    if let Some(dragged_piece) = q_dragged_pieces.iter().next() {
        let (mut board_sprite, board_tile) = q_board_tiles
            .iter_mut()
            .find(|(_, board_tile)| board_tile.0 == dragged_piece.square())
            .expect("Every square should exist.");
        board_sprite.color.set_b(1.0);
    }
}
