use bevy::{
    color::palettes::css::*,
    math::bounding::{Aabb3d, IntersectsVolume},
    prelude::*,
    utils::HashMap,
};
use rand::Rng;

use crate::utils::{self, freecam::FreeCamera};

pub fn run() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(utils::freecam::FreeCameraPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, utils::toggle_cursor_grab_with_esc)
        .run();
}

struct Room {
    room_type: RoomType,
}

impl Room {
    fn size(&self) -> Vec3 {
        self.room_type.size()
    }
}

enum RoomType {
    Loot,
    Spawn,
    Boss,
}

impl RoomType {
    fn size(&self) -> Vec3 {
        match self {
            RoomType::Loot => Vec3::new(2.0, 2.0, 2.0),
            RoomType::Spawn => Vec3::new(3.0, 2.0, 3.0),
            RoomType::Boss => Vec3::new(3.0, 2.0, 3.0),
        }
    }
}

pub struct Dungeon {
    rooms: HashMap<IVec3, Room>,
}

impl Dungeon {
    pub fn render(&self, root: Vec3, mut commands: Commands, asset_server: &AssetServer) {
        for (pos, room) in &self.rooms {
            let translation = root + pos.as_vec3();
            let size = room.room_type.size();
            let mesh = Cuboid::new(size.x, size.y, size.z);
            let color = match room.room_type {
                RoomType::Loot => YELLOW,
                RoomType::Spawn => GREEN,
                RoomType::Boss => RED,
            };
            commands.spawn((
                Mesh3d(asset_server.add(mesh.mesh().build())),
                MeshMaterial3d(asset_server.add(StandardMaterial::from_color(color))),
                Transform::from_translation(translation),
            ));
        }
    }
}

pub struct DungeonConfig {
    initial_radius: Vec3,
	min_separation: Vec3,
    n_rooms: usize,
}

pub fn generate_dungeon(cfg: &DungeonConfig) -> Dungeon {
    let mut rooms: Vec<(Vec3, Room)> = vec![];
    let mut rng = rand::thread_rng();

    for _ in 0..cfg.n_rooms {
        let x = rng.gen_range(-cfg.initial_radius.x..cfg.initial_radius.x);
        let y = rng.gen_range(-cfg.initial_radius.y..cfg.initial_radius.y);
        let z = rng.gen_range(-cfg.initial_radius.z..cfg.initial_radius.z);
        let pos = Vec3::new(x, y, z);
        let room_type = match rng.gen_range(0..=2) {
            0 => RoomType::Loot,
            1 => RoomType::Spawn,
            2 => RoomType::Boss,
            _ => unreachable!(),
        };
        rooms.push((pos, Room { room_type }));
    }

    let max_iter = 100;
    for iter in 0..=max_iter {
        let mut seen_collision = false;
        for i in 0..rooms.len() {
            for j in 0..rooms.len() {
                if i == j {
                    continue;
                }

                let (pos_i, room_i) = &rooms[i];
                let (pos_j, room_j) = &rooms[j];
                let pos_i = *pos_i;
                let pos_j = *pos_j;

                let i_aabb = Aabb3d::new(pos_i, (room_i.size() + cfg.min_separation) / 2.0);
                let j_aabb = Aabb3d::new(pos_j, (room_j.size() + cfg.min_separation) / 2.0);

                if i_aabb.intersects(&j_aabb) {
                    seen_collision = true;
                    rooms[i].0 += pos_i - pos_j;
                    rooms[j].0 += pos_j - pos_i;
                }
            }
        }

        if !seen_collision {
            info!("dungeon generated after {} iterations", iter);
            break;
        }
        if iter == max_iter {
            panic!("failed dungeon gen, max iter reached");
        }
    }

    let mut room_map: HashMap<IVec3, Room> = HashMap::with_capacity(rooms.len());
    for (pos, room) in rooms {
        room_map.insert(pos.as_ivec3(), room);
    }

    Dungeon { rooms: room_map }
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    // Spawn the camera.
    commands.spawn((
        Camera3d::default(),
        FreeCamera::new(4.0),
        Transform::from_translation(Vec3::splat(6.0)).looking_at(Vec3::new(0., 1., 0.), Vec3::Y),
    ));

    // Spawn the light.
    commands.spawn((
        PointLight {
            intensity: 10_000_000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(-4.0, 8.0, 13.0),
    ));

    let config = DungeonConfig {
        initial_radius: Vec3::splat(10.0),
		min_separation: Vec3::splat(2.0),
        n_rooms: 40,
    };
    generate_dungeon(&config).render(Vec3::ZERO, commands, &asset_server);
}
