// todo("windows"): remove
#![cfg_attr(windows, allow(dead_code))]

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    AtlasTextureId, AtlasTile, Background, Bounds, ContentMask, Corners, Edges, Hsla, Pixels,
    Point, Radians, ScaledPixels, Size, bounds_tree::BoundsTree, point,
};
use std::{
    fmt::Debug,
    iter::Peekable,
    ops::{Add, Range, Sub},
    slice,
};

#[allow(non_camel_case_types, unused)]
#[expect(missing_docs)]
pub type PathVertex_ScaledPixels = PathVertex<ScaledPixels>;

#[expect(missing_docs)]
pub type DrawOrder = u32;

#[derive(Default)]
#[expect(missing_docs)]
pub struct Scene {
    pub(crate) paint_operations: Vec<PaintOperation>,
    primitive_bounds: BoundsTree<ScaledPixels>,
    layer_stack: Vec<DrawOrder>,
    pub shadows: Vec<Shadow>,
    pub quads: Vec<Quad>,
    pub paths: Vec<Path<ScaledPixels>>,
    pub underlines: Vec<Underline>,
    pub monochrome_sprites: Vec<MonochromeSprite>,
    pub subpixel_sprites: Vec<SubpixelSprite>,
    pub polychrome_sprites: Vec<PolychromeSprite>,
    pub surfaces: Vec<PaintSurface>,
    pub blur_rects: Vec<BlurRect>,
    pub lens_rects: Vec<LensRect>,
    pub mirror_rects: Vec<MirrorRect>,
    /// Per-shape data for `LensRect`s that represent a multi-shape union.
    /// Each `LensRect` points at a `shape_offset..shape_offset+shape_count`
    /// slice of this vec. Shapes are never reordered after insertion, so the
    /// offsets stored in `lens_rects` survive `finish()`'s sort.
    pub lens_shapes: Vec<LensShape>,
}

#[expect(missing_docs)]
impl Scene {
    pub fn clear(&mut self) {
        self.paint_operations.clear();
        self.primitive_bounds.clear();
        self.layer_stack.clear();
        self.paths.clear();
        self.shadows.clear();
        self.quads.clear();
        self.underlines.clear();
        self.monochrome_sprites.clear();
        self.subpixel_sprites.clear();
        self.polychrome_sprites.clear();
        self.surfaces.clear();
        self.blur_rects.clear();
        self.lens_rects.clear();
        self.mirror_rects.clear();
        self.lens_shapes.clear();
    }

    pub fn len(&self) -> usize {
        self.paint_operations.len()
    }

    pub fn push_layer(&mut self, bounds: Bounds<ScaledPixels>) {
        let order = self.primitive_bounds.insert(bounds);
        self.layer_stack.push(order);
        self.paint_operations
            .push(PaintOperation::StartLayer(bounds));
    }

    pub fn pop_layer(&mut self) {
        self.layer_stack.pop();
        self.paint_operations.push(PaintOperation::EndLayer);
    }

    pub fn insert_primitive(&mut self, primitive: impl Into<Primitive>) {
        let mut primitive = primitive.into();
        let clipped_bounds = primitive
            .bounds()
            .intersect(&primitive.content_mask().bounds);

        if clipped_bounds.is_empty() {
            return;
        }

        let order = self
            .layer_stack
            .last()
            .copied()
            .unwrap_or_else(|| self.primitive_bounds.insert(clipped_bounds));
        match &mut primitive {
            Primitive::Shadow(shadow) => {
                shadow.order = order;
                self.shadows.push(*shadow);
            }
            Primitive::Quad(quad) => {
                quad.order = order;
                self.quads.push(*quad);
            }
            Primitive::Path(path) => {
                path.order = order;
                path.id = PathId(self.paths.len());
                self.paths.push(path.clone());
            }
            Primitive::Underline(underline) => {
                underline.order = order;
                self.underlines.push(*underline);
            }
            Primitive::MonochromeSprite(sprite) => {
                sprite.order = order;
                self.monochrome_sprites.push(*sprite);
            }
            Primitive::SubpixelSprite(sprite) => {
                sprite.order = order;
                self.subpixel_sprites.push(*sprite);
            }
            Primitive::PolychromeSprite(sprite) => {
                sprite.order = order;
                self.polychrome_sprites.push(*sprite);
            }
            Primitive::Surface(surface) => {
                surface.order = order;
                self.surfaces.push(surface.clone());
            }
            Primitive::BlurRect(blur) => {
                blur.order = order;
                self.blur_rects.push(*blur);
            }
            Primitive::LensRect(lens) => {
                lens.rect.order = order;
                lens.rect.shape_offset = self.lens_shapes.len() as u32;
                lens.rect.shape_count = lens.shapes.len() as u32;
                self.lens_shapes.extend_from_slice(&lens.shapes);
                self.lens_rects.push(lens.rect);
            }
            Primitive::MirrorRect(mirror) => {
                mirror.order = order;
                self.mirror_rects.push(*mirror);
            }
        }
        self.paint_operations
            .push(PaintOperation::Primitive(primitive));
    }

    pub fn replay(&mut self, range: Range<usize>, prev_scene: &Scene) {
        for operation in &prev_scene.paint_operations[range] {
            match operation {
                PaintOperation::Primitive(primitive) => self.insert_primitive(primitive.clone()),
                PaintOperation::StartLayer(bounds) => self.push_layer(*bounds),
                PaintOperation::EndLayer => self.pop_layer(),
            }
        }
    }

    pub fn finish(&mut self) {
        self.shadows.sort_by_key(|shadow| shadow.order);
        self.quads.sort_by_key(|quad| quad.order);
        self.paths.sort_by_key(|path| path.order);
        self.underlines.sort_by_key(|underline| underline.order);
        self.monochrome_sprites
            .sort_by_key(|sprite| (sprite.order, sprite.tile.tile_id));
        self.subpixel_sprites
            .sort_by_key(|sprite| (sprite.order, sprite.tile.tile_id));
        self.polychrome_sprites
            .sort_by_key(|sprite| (sprite.order, sprite.tile.tile_id));
        self.surfaces.sort_by_key(|surface| surface.order);
        self.blur_rects.sort_by_key(|blur| blur.order);
        self.lens_rects.sort_by_key(|lens| lens.order);
        self.mirror_rects.sort_by_key(|mirror| mirror.order);
    }

    /// Maximum `kernel_levels` requested by any `BlurRect`, `LensRect`, or
    /// `MirrorRect` in the scene, clamped to `1..=5`. Used by renderers
    /// to size the dual-Kawase snapshot chain for the frame. Returns 0
    /// when no blur-consuming primitive is present.
    pub fn max_blur_kernel_levels(&self) -> u32 {
        let blur = self
            .blur_rects
            .iter()
            .map(|b| b.kernel_levels)
            .max()
            .unwrap_or(0);
        let lens = self
            .lens_rects
            .iter()
            .map(|l| l.kernel_levels)
            .max()
            .unwrap_or(0);
        let mirror = self
            .mirror_rects
            .iter()
            .filter(|m| m.blur_radius.0 > 0.0 || m.blur_radius_end.0 > 0.0)
            .map(|m| m.kernel_levels)
            .max()
            .unwrap_or(0);
        blur.max(lens).max(mirror).min(5)
    }

    /// Maximum `blur_radius` requested by any `BlurRect`, `LensRect`, or
    /// `MirrorRect` in the scene. For gradient blurs, considers both
    /// endpoints (`radius` and `radius_end`) — the bigger of the two
    /// determines how much blurring the Kawase chain must be able to
    /// produce. Zero when no blur-consuming primitive is present.
    pub fn max_blur_radius(&self) -> ScaledPixels {
        let blur = self
            .blur_rects
            .iter()
            .map(|b| b.blur_radius.0.max(b.blur_radius_end.0))
            .fold(0.0_f32, f32::max);
        let lens = self
            .lens_rects
            .iter()
            .map(|l| l.blur_radius.0.max(l.blur_radius_end.0))
            .fold(0.0_f32, f32::max);
        let mirror = self
            .mirror_rects
            .iter()
            .map(|m| m.blur_radius.0.max(m.blur_radius_end.0))
            .fold(0.0_f32, f32::max);
        ScaledPixels(blur.max(lens).max(mirror))
    }

    #[cfg_attr(
        all(
            any(target_os = "linux", target_os = "freebsd"),
            not(any(feature = "x11", feature = "wayland"))
        ),
        allow(dead_code)
    )]
    pub fn batches(&self) -> impl Iterator<Item = PrimitiveBatch> + '_ {
        BatchIterator {
            shadows_start: 0,
            shadows_iter: self.shadows.iter().peekable(),
            quads_start: 0,
            quads_iter: self.quads.iter().peekable(),
            paths_start: 0,
            paths_iter: self.paths.iter().peekable(),
            underlines_start: 0,
            underlines_iter: self.underlines.iter().peekable(),
            monochrome_sprites_start: 0,
            monochrome_sprites_iter: self.monochrome_sprites.iter().peekable(),
            subpixel_sprites_start: 0,
            subpixel_sprites_iter: self.subpixel_sprites.iter().peekable(),
            polychrome_sprites_start: 0,
            polychrome_sprites_iter: self.polychrome_sprites.iter().peekable(),
            surfaces_start: 0,
            surfaces_iter: self.surfaces.iter().peekable(),
            blur_rects_start: 0,
            blur_rects_iter: self.blur_rects.iter().peekable(),
            lens_rects_start: 0,
            lens_rects_iter: self.lens_rects.iter().peekable(),
            mirror_rects_start: 0,
            mirror_rects_iter: self.mirror_rects.iter().peekable(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Default)]
#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
pub(crate) enum PrimitiveKind {
    Shadow,
    #[default]
    Quad,
    Path,
    Underline,
    MonochromeSprite,
    SubpixelSprite,
    PolychromeSprite,
    Surface,
    BlurRect,
    LensRect,
    MirrorRect,
}

pub(crate) enum PaintOperation {
    Primitive(Primitive),
    StartLayer(Bounds<ScaledPixels>),
    EndLayer,
}

#[derive(Clone)]
#[expect(missing_docs)]
pub enum Primitive {
    Shadow(Shadow),
    Quad(Quad),
    Path(Path<ScaledPixels>),
    Underline(Underline),
    MonochromeSprite(MonochromeSprite),
    SubpixelSprite(SubpixelSprite),
    PolychromeSprite(PolychromeSprite),
    Surface(PaintSurface),
    BlurRect(BlurRect),
    LensRect(LensPrimitive),
    MirrorRect(MirrorRect),
}

/// A `LensRect` header paired with its per-shape union inputs. Round-trips
/// through `PaintOperation::Primitive` so `Scene::replay` can repopulate
/// `Scene::lens_shapes` with fresh offsets.
#[derive(Clone, Debug)]
#[expect(missing_docs)]
pub struct LensPrimitive {
    pub rect: LensRect,
    pub shapes: Vec<LensShape>,
}

#[expect(missing_docs)]
impl Primitive {
    pub fn bounds(&self) -> &Bounds<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &shadow.bounds,
            Primitive::Quad(quad) => &quad.bounds,
            Primitive::Path(path) => &path.bounds,
            Primitive::Underline(underline) => &underline.bounds,
            Primitive::MonochromeSprite(sprite) => &sprite.bounds,
            Primitive::SubpixelSprite(sprite) => &sprite.bounds,
            Primitive::PolychromeSprite(sprite) => &sprite.bounds,
            Primitive::Surface(surface) => &surface.bounds,
            Primitive::BlurRect(blur) => &blur.bounds,
            Primitive::LensRect(lens) => &lens.rect.bounds,
            Primitive::MirrorRect(mirror) => &mirror.bounds,
        }
    }

    pub fn content_mask(&self) -> &ContentMask<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &shadow.content_mask,
            Primitive::Quad(quad) => &quad.content_mask,
            Primitive::Path(path) => &path.content_mask,
            Primitive::Underline(underline) => &underline.content_mask,
            Primitive::MonochromeSprite(sprite) => &sprite.content_mask,
            Primitive::SubpixelSprite(sprite) => &sprite.content_mask,
            Primitive::PolychromeSprite(sprite) => &sprite.content_mask,
            Primitive::Surface(surface) => &surface.content_mask,
            Primitive::BlurRect(blur) => &blur.content_mask,
            Primitive::LensRect(lens) => &lens.rect.content_mask,
            Primitive::MirrorRect(mirror) => &mirror.content_mask,
        }
    }
}

#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
struct BatchIterator<'a> {
    shadows_start: usize,
    shadows_iter: Peekable<slice::Iter<'a, Shadow>>,
    quads_start: usize,
    quads_iter: Peekable<slice::Iter<'a, Quad>>,
    paths_start: usize,
    paths_iter: Peekable<slice::Iter<'a, Path<ScaledPixels>>>,
    underlines_start: usize,
    underlines_iter: Peekable<slice::Iter<'a, Underline>>,
    monochrome_sprites_start: usize,
    monochrome_sprites_iter: Peekable<slice::Iter<'a, MonochromeSprite>>,
    subpixel_sprites_start: usize,
    subpixel_sprites_iter: Peekable<slice::Iter<'a, SubpixelSprite>>,
    polychrome_sprites_start: usize,
    polychrome_sprites_iter: Peekable<slice::Iter<'a, PolychromeSprite>>,
    surfaces_start: usize,
    surfaces_iter: Peekable<slice::Iter<'a, PaintSurface>>,
    blur_rects_start: usize,
    blur_rects_iter: Peekable<slice::Iter<'a, BlurRect>>,
    lens_rects_start: usize,
    lens_rects_iter: Peekable<slice::Iter<'a, LensRect>>,
    mirror_rects_start: usize,
    mirror_rects_iter: Peekable<slice::Iter<'a, MirrorRect>>,
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = PrimitiveBatch;

    fn next(&mut self) -> Option<Self::Item> {
        let mut orders_and_kinds = [
            (
                self.shadows_iter.peek().map(|s| s.order),
                PrimitiveKind::Shadow,
            ),
            (self.quads_iter.peek().map(|q| q.order), PrimitiveKind::Quad),
            (self.paths_iter.peek().map(|q| q.order), PrimitiveKind::Path),
            (
                self.underlines_iter.peek().map(|u| u.order),
                PrimitiveKind::Underline,
            ),
            (
                self.monochrome_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::MonochromeSprite,
            ),
            (
                self.subpixel_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::SubpixelSprite,
            ),
            (
                self.polychrome_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::PolychromeSprite,
            ),
            (
                self.surfaces_iter.peek().map(|s| s.order),
                PrimitiveKind::Surface,
            ),
            (
                self.blur_rects_iter.peek().map(|b| b.order),
                PrimitiveKind::BlurRect,
            ),
            (
                self.lens_rects_iter.peek().map(|l| l.order),
                PrimitiveKind::LensRect,
            ),
            (
                self.mirror_rects_iter.peek().map(|m| m.order),
                PrimitiveKind::MirrorRect,
            ),
        ];
        orders_and_kinds.sort_by_key(|(order, kind)| (order.unwrap_or(u32::MAX), *kind));

        let first = orders_and_kinds[0];
        let second = orders_and_kinds[1];
        let (batch_kind, max_order_and_kind) = if first.0.is_some() {
            (first.1, (second.0.unwrap_or(u32::MAX), second.1))
        } else {
            return None;
        };

        match batch_kind {
            PrimitiveKind::Shadow => {
                let shadows_start = self.shadows_start;
                let mut shadows_end = shadows_start + 1;
                self.shadows_iter.next();
                while self
                    .shadows_iter
                    .next_if(|shadow| (shadow.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    shadows_end += 1;
                }
                self.shadows_start = shadows_end;
                Some(PrimitiveBatch::Shadows(shadows_start..shadows_end))
            }
            PrimitiveKind::Quad => {
                let quads_start = self.quads_start;
                let mut quads_end = quads_start + 1;
                self.quads_iter.next();
                while self
                    .quads_iter
                    .next_if(|quad| (quad.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    quads_end += 1;
                }
                self.quads_start = quads_end;
                Some(PrimitiveBatch::Quads(quads_start..quads_end))
            }
            PrimitiveKind::Path => {
                let paths_start = self.paths_start;
                let mut paths_end = paths_start + 1;
                self.paths_iter.next();
                while self
                    .paths_iter
                    .next_if(|path| (path.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    paths_end += 1;
                }
                self.paths_start = paths_end;
                Some(PrimitiveBatch::Paths(paths_start..paths_end))
            }
            PrimitiveKind::Underline => {
                let underlines_start = self.underlines_start;
                let mut underlines_end = underlines_start + 1;
                self.underlines_iter.next();
                while self
                    .underlines_iter
                    .next_if(|underline| (underline.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    underlines_end += 1;
                }
                self.underlines_start = underlines_end;
                Some(PrimitiveBatch::Underlines(underlines_start..underlines_end))
            }
            PrimitiveKind::MonochromeSprite => {
                let texture_id = self.monochrome_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.monochrome_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.monochrome_sprites_iter.next();
                while self
                    .monochrome_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.monochrome_sprites_start = sprites_end;
                Some(PrimitiveBatch::MonochromeSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                })
            }
            PrimitiveKind::SubpixelSprite => {
                let texture_id = self.subpixel_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.subpixel_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.subpixel_sprites_iter.next();
                while self
                    .subpixel_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.subpixel_sprites_start = sprites_end;
                Some(PrimitiveBatch::SubpixelSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                })
            }
            PrimitiveKind::PolychromeSprite => {
                let texture_id = self.polychrome_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.polychrome_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.polychrome_sprites_iter.next();
                while self
                    .polychrome_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.polychrome_sprites_start = sprites_end;
                Some(PrimitiveBatch::PolychromeSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                })
            }
            PrimitiveKind::Surface => {
                let surfaces_start = self.surfaces_start;
                let mut surfaces_end = surfaces_start + 1;
                self.surfaces_iter.next();
                while self
                    .surfaces_iter
                    .next_if(|surface| (surface.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    surfaces_end += 1;
                }
                self.surfaces_start = surfaces_end;
                Some(PrimitiveBatch::Surfaces(surfaces_start..surfaces_end))
            }
            PrimitiveKind::BlurRect => {
                let blur_rects_start = self.blur_rects_start;
                let mut blur_rects_end = blur_rects_start + 1;
                self.blur_rects_iter.next();
                while self
                    .blur_rects_iter
                    .next_if(|blur| (blur.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    blur_rects_end += 1;
                }
                self.blur_rects_start = blur_rects_end;
                Some(PrimitiveBatch::BlurRects(blur_rects_start..blur_rects_end))
            }
            PrimitiveKind::LensRect => {
                let lens_rects_start = self.lens_rects_start;
                let mut lens_rects_end = lens_rects_start + 1;
                self.lens_rects_iter.next();
                while self
                    .lens_rects_iter
                    .next_if(|lens| (lens.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    lens_rects_end += 1;
                }
                self.lens_rects_start = lens_rects_end;
                Some(PrimitiveBatch::LensRects(lens_rects_start..lens_rects_end))
            }
            PrimitiveKind::MirrorRect => {
                let mirror_rects_start = self.mirror_rects_start;
                let mut mirror_rects_end = mirror_rects_start + 1;
                self.mirror_rects_iter.next();
                while self
                    .mirror_rects_iter
                    .next_if(|mirror| (mirror.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    mirror_rects_end += 1;
                }
                self.mirror_rects_start = mirror_rects_end;
                Some(PrimitiveBatch::MirrorRects(
                    mirror_rects_start..mirror_rects_end,
                ))
            }
        }
    }
}

#[derive(Debug)]
#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
#[allow(missing_docs)]
pub enum PrimitiveBatch {
    Shadows(Range<usize>),
    Quads(Range<usize>),
    Paths(Range<usize>),
    Underlines(Range<usize>),
    MonochromeSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
    },
    #[cfg_attr(target_os = "macos", allow(dead_code))]
    SubpixelSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
    },
    PolychromeSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
    },
    Surfaces(Range<usize>),
    BlurRects(Range<usize>),
    LensRects(Range<usize>),
    MirrorRects(Range<usize>),
}

#[derive(Default, Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Quad {
    pub order: DrawOrder,
    pub border_style: BorderStyle,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub background: Background,
    pub border_color: Hsla,
    pub corner_radii: Corners<ScaledPixels>,
    pub border_widths: Edges<ScaledPixels>,
}

impl From<Quad> for Primitive {
    fn from(quad: Quad) -> Self {
        Primitive::Quad(quad)
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Underline {
    pub order: DrawOrder,
    pub pad: u32, // align to 8 bytes
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub thickness: ScaledPixels,
    pub wavy: u32,
}

impl From<Underline> for Primitive {
    fn from(underline: Underline) -> Self {
        Primitive::Underline(underline)
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Shadow {
    pub order: DrawOrder,
    pub blur_radius: ScaledPixels,
    pub bounds: Bounds<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
}

impl From<Shadow> for Primitive {
    fn from(shadow: Shadow) -> Self {
        Primitive::Shadow(shadow)
    }
}

/// A rounded rectangle that samples the framebuffer behind it, runs a
/// dual-Kawase blur, and composites the result with a tint. Used for
/// backdrop-blurred materials like Apple's Liquid Glass.
///
/// Supports uniform or linear-gradient blur strength: when
/// `gradient_direction != [0, 0]`, the fragment shader interpolates
/// `blur_radius_start -> blur_radius_end` along that axis and mixes between
/// the un-blurred framebuffer and the maximum-blur output. Used for HIG
/// scroll-edge fades.
///
/// Unlike `Quad` / `Shadow` / etc., a `BlurRect` is not drawn with an
/// instanced pipeline — each one breaks the current render pass so the
/// framebuffer can be sampled, then runs its own post-process chain before
/// the main pass resumes. Overuse is therefore expensive; prefer a handful
/// per frame.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
#[expect(missing_docs)]
pub struct BlurRect {
    pub order: DrawOrder,
    pub kernel_levels: u32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    /// Starting blur radius (in device pixels). For uniform blur, equals
    /// `blur_radius_end`. The max of the two is what sizes the Kawase chain.
    pub blur_radius: ScaledPixels,
    /// Ending blur radius (in device pixels) for linear-gradient blur.
    pub blur_radius_end: ScaledPixels,
    /// Normalized gradient direction. `[0.0, 0.0]` means uniform blur.
    pub gradient_direction: [f32; 2],
    pub tint: Hsla,
    /// Pads the stride to 96 bytes so the WGSL `array<BlurRect>` matches
    /// the Rust side (8-byte alignment required for the nested `vec2<f32>`
    /// Bounds).
    pub _pad_end: [f32; 2],
}

const _: () = assert!(std::mem::size_of::<BlurRect>() == 96);

impl From<BlurRect> for Primitive {
    fn from(blur: BlurRect) -> Self {
        Primitive::BlurRect(blur)
    }
}

/// A rounded-rectangle group with a full Liquid Glass composite: backdrop
/// blur plus parabolic refraction, chromatic aberration, and a directional
/// Fresnel edge highlight. `bounds` is the union AABB of the shapes this
/// primitive covers, expanded by `edge_blend_distance` on all sides so the
/// merge field has room to settle. Per-shape `corner_radii` and `bounds`
/// live on separate `LensShape` entries at
/// `Scene::lens_shapes[shape_offset..shape_offset + shape_count]`.
///
/// Supports uniform or linear-gradient blur strength the same way
/// [`BlurRect`] does: `gradient_direction != [0, 0]` enables a per-pixel
/// interpolation between `blur_radius..blur_radius_end`.
///
/// The field layout is kept in lockstep with the WGSL / MSL `LensRect`
/// struct and padded to a 112-byte stride.
///
/// Same caveat as `BlurRect`: each `LensRect` forces a render-pass break.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
#[expect(missing_docs)]
pub struct LensRect {
    pub order: DrawOrder,
    pub kernel_levels: u32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    /// Index into `Scene::lens_shapes` of this lens's first shape.
    pub shape_offset: u32,
    /// Number of shapes in this lens (1..=`MAX_LENS_SHAPES_PER_GROUP`).
    pub shape_count: u32,
    /// Smooth-min radius, in pixels, used when merging overlapping shapes
    /// into a single SDF. 0 disables merging (hard min).
    pub edge_blend_distance: ScaledPixels,
    /// Starting blur radius (in device pixels). For uniform blur, equals
    /// `blur_radius_end`.
    pub blur_radius: ScaledPixels,
    /// Normalized gradient direction. `[0.0, 0.0]` means uniform blur.
    pub gradient_direction: [f32; 2],
    /// Ending blur radius (in device pixels) for linear-gradient blur.
    pub blur_radius_end: ScaledPixels,
    pub refraction: f32,
    pub depth: f32,
    pub dispersion: f32,
    pub splay: ScaledPixels,
    pub light_angle_radians: f32,
    pub light_intensity: f32,
    pub tint: Hsla,
    // Pads the struct to 112 bytes so its WGSL `array<LensRect>` stride
    // (rounded to 8-byte alignment for the nested `Bounds` `vec2<f32>`)
    // matches the Rust storage-buffer stride.
    pub pad2: f32,
}

const _: () = assert!(std::mem::size_of::<LensRect>() == 112);

impl From<LensPrimitive> for Primitive {
    fn from(lens: LensPrimitive) -> Self {
        Primitive::LensRect(lens)
    }
}

/// One shape in a multi-shape `LensRect` union. The lens fragment shader
/// computes a signed distance to each shape, combines them with a smooth-min
/// weighted by `LensRect::edge_blend_distance`, and refracts against the
/// resulting field — letting overlapping shapes visually morph into one.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
#[expect(missing_docs)]
pub struct LensShape {
    pub bounds: Bounds<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    /// Pads the struct to 48 bytes so its WGSL `array<LensShape>` stride
    /// (rounded to 8-byte alignment for the nested `Bounds` `vec2<f32>`)
    /// matches the Rust storage-buffer stride.
    pub _pad: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<LensShape>() == 48);

/// Bit flag enabling a horizontal (left/right) flip in
/// [`MirrorRect::mirror_axis`].
pub const MIRROR_AXIS_HORIZONTAL: u32 = 1 << 0;
/// Bit flag enabling a vertical (top/bottom) flip in
/// [`MirrorRect::mirror_axis`]. Combine with
/// [`MIRROR_AXIS_HORIZONTAL`] to produce a 180° rotation.
pub const MIRROR_AXIS_VERTICAL: u32 = 1 << 1;

/// A rounded rectangle that samples a region of the framebuffer, optionally
/// flipped across one or both axes, and composites the result back at a
/// different location. Used for SwiftUI's `backgroundExtensionEffect()` —
/// extend an image's edge into a decorative out-of-frame mirror, optionally
/// blurred with the scene's Kawase chain.
///
/// Supports the same uniform-or-gradient blur as [`BlurRect`] /
/// [`LensRect`]. `blur_radius == 0 && blur_radius_end == 0` samples the
/// un-blurred framebuffer snapshot directly (no blur).
///
/// Like `BlurRect` and `LensRect`, each mirror primitive forces a
/// render-pass break so the framebuffer can be sampled.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
#[expect(missing_docs)]
pub struct MirrorRect {
    pub order: DrawOrder,
    pub kernel_levels: u32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub source_bounds: Bounds<ScaledPixels>,
    /// Bitmask of `MIRROR_AXIS_*` flags. `0` disables flipping; the
    /// primitive then acts as a relocated copy of `source_bounds`.
    pub mirror_axis: u32,
    /// `1` clips the mirrored sample against the rounded-rect SDF so
    /// out-of-corner pixels go transparent. `0` lets the mirrored
    /// rectangle bleed into the AA band.
    pub clip_to_destination: u32,
    pub blur_radius: ScaledPixels,
    pub blur_radius_end: ScaledPixels,
    pub gradient_direction: [f32; 2],
    pub tint: Hsla,
}

const _: () = assert!(std::mem::size_of::<MirrorRect>() == 112);

impl From<MirrorRect> for Primitive {
    fn from(mirror: MirrorRect) -> Self {
        Primitive::MirrorRect(mirror)
    }
}

/// The style of a border.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[repr(C)]
pub enum BorderStyle {
    /// A solid border.
    #[default]
    Solid = 0,
    /// A dashed border.
    Dashed = 1,
}

/// A data type representing a 2 dimensional transformation that can be applied to an element.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct TransformationMatrix {
    /// 2x2 matrix containing rotation and scale,
    /// stored row-major
    pub rotation_scale: [[f32; 2]; 2],
    /// translation vector
    pub translation: [f32; 2],
}

impl Eq for TransformationMatrix {}

impl TransformationMatrix {
    /// The unit matrix, has no effect.
    pub fn unit() -> Self {
        Self {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [0.0, 0.0],
        }
    }

    /// Move the origin by a given point
    pub fn translate(mut self, point: Point<ScaledPixels>) -> Self {
        self.compose(Self {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [point.x.0, point.y.0],
        })
    }

    /// Clockwise rotation in radians around the origin
    pub fn rotate(self, angle: Radians) -> Self {
        self.compose(Self {
            rotation_scale: [
                [angle.0.cos(), -angle.0.sin()],
                [angle.0.sin(), angle.0.cos()],
            ],
            translation: [0.0, 0.0],
        })
    }

    /// Scale around the origin
    pub fn scale(self, size: Size<f32>) -> Self {
        self.compose(Self {
            rotation_scale: [[size.width, 0.0], [0.0, size.height]],
            translation: [0.0, 0.0],
        })
    }

    /// Perform matrix multiplication with another transformation
    /// to produce a new transformation that is the result of
    /// applying both transformations: first, `other`, then `self`.
    #[inline]
    pub fn compose(self, other: TransformationMatrix) -> TransformationMatrix {
        if other == Self::unit() {
            return self;
        }
        // Perform matrix multiplication
        TransformationMatrix {
            rotation_scale: [
                [
                    self.rotation_scale[0][0] * other.rotation_scale[0][0]
                        + self.rotation_scale[0][1] * other.rotation_scale[1][0],
                    self.rotation_scale[0][0] * other.rotation_scale[0][1]
                        + self.rotation_scale[0][1] * other.rotation_scale[1][1],
                ],
                [
                    self.rotation_scale[1][0] * other.rotation_scale[0][0]
                        + self.rotation_scale[1][1] * other.rotation_scale[1][0],
                    self.rotation_scale[1][0] * other.rotation_scale[0][1]
                        + self.rotation_scale[1][1] * other.rotation_scale[1][1],
                ],
            ],
            translation: [
                self.translation[0]
                    + self.rotation_scale[0][0] * other.translation[0]
                    + self.rotation_scale[0][1] * other.translation[1],
                self.translation[1]
                    + self.rotation_scale[1][0] * other.translation[0]
                    + self.rotation_scale[1][1] * other.translation[1],
            ],
        }
    }

    /// Apply transformation to a point, mainly useful for debugging
    pub fn apply(&self, point: Point<Pixels>) -> Point<Pixels> {
        let input = [point.x.0, point.y.0];
        let mut output = self.translation;
        for (i, output_cell) in output.iter_mut().enumerate() {
            for (k, input_cell) in input.iter().enumerate() {
                *output_cell += self.rotation_scale[i][k] * *input_cell;
            }
        }
        Point::new(output[0].into(), output[1].into())
    }
}

impl Default for TransformationMatrix {
    fn default() -> Self {
        Self::unit()
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct MonochromeSprite {
    pub order: DrawOrder,
    pub pad: u32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub tile: AtlasTile,
    pub transformation: TransformationMatrix,
}

impl From<MonochromeSprite> for Primitive {
    fn from(sprite: MonochromeSprite) -> Self {
        Primitive::MonochromeSprite(sprite)
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct SubpixelSprite {
    pub order: DrawOrder,
    pub pad: u32, // align to 8 bytes
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub tile: AtlasTile,
    pub transformation: TransformationMatrix,
}

impl From<SubpixelSprite> for Primitive {
    fn from(sprite: SubpixelSprite) -> Self {
        Primitive::SubpixelSprite(sprite)
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PolychromeSprite {
    pub order: DrawOrder,
    pub pad: u32,
    pub grayscale: bool,
    pub opacity: f32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub tile: AtlasTile,
}

impl From<PolychromeSprite> for Primitive {
    fn from(sprite: PolychromeSprite) -> Self {
        Primitive::PolychromeSprite(sprite)
    }
}

#[derive(Clone, Debug)]
#[allow(missing_docs)]
pub struct PaintSurface {
    pub order: DrawOrder,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    #[cfg(target_os = "macos")]
    pub image_buffer: core_video::pixel_buffer::CVPixelBuffer,
}

impl From<PaintSurface> for Primitive {
    fn from(surface: PaintSurface) -> Self {
        Primitive::Surface(surface)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[expect(missing_docs)]
pub struct PathId(pub usize);

/// A line made up of a series of vertices and control points.
#[derive(Clone, Debug)]
#[expect(missing_docs)]
pub struct Path<P: Clone + Debug + Default + PartialEq> {
    pub id: PathId,
    pub order: DrawOrder,
    pub bounds: Bounds<P>,
    pub content_mask: ContentMask<P>,
    pub vertices: Vec<PathVertex<P>>,
    pub color: Background,
    start: Point<P>,
    current: Point<P>,
    contour_count: usize,
}

impl Path<Pixels> {
    /// Create a new path with the given starting point.
    pub fn new(start: Point<Pixels>) -> Self {
        Self {
            id: PathId(0),
            order: DrawOrder::default(),
            vertices: Vec::new(),
            start,
            current: start,
            bounds: Bounds {
                origin: start,
                size: Default::default(),
            },
            content_mask: Default::default(),
            color: Default::default(),
            contour_count: 0,
        }
    }

    /// Scale this path by the given factor.
    pub fn scale(&self, factor: f32) -> Path<ScaledPixels> {
        Path {
            id: self.id,
            order: self.order,
            bounds: self.bounds.scale(factor),
            content_mask: self.content_mask.scale(factor),
            vertices: self
                .vertices
                .iter()
                .map(|vertex| vertex.scale(factor))
                .collect(),
            start: self.start.map(|start| start.scale(factor)),
            current: self.current.scale(factor),
            contour_count: self.contour_count,
            color: self.color,
        }
    }

    /// Move the start, current point to the given point.
    pub fn move_to(&mut self, to: Point<Pixels>) {
        self.contour_count += 1;
        self.start = to;
        self.current = to;
    }

    /// Draw a straight line from the current point to the given point.
    pub fn line_to(&mut self, to: Point<Pixels>) {
        self.contour_count += 1;
        if self.contour_count > 1 {
            self.push_triangle(
                (self.start, self.current, to),
                (point(0., 1.), point(0., 1.), point(0., 1.)),
            );
        }
        self.current = to;
    }

    /// Draw a curve from the current point to the given point, using the given control point.
    pub fn curve_to(&mut self, to: Point<Pixels>, ctrl: Point<Pixels>) {
        self.contour_count += 1;
        if self.contour_count > 1 {
            self.push_triangle(
                (self.start, self.current, to),
                (point(0., 1.), point(0., 1.), point(0., 1.)),
            );
        }

        self.push_triangle(
            (self.current, ctrl, to),
            (point(0., 0.), point(0.5, 0.), point(1., 1.)),
        );
        self.current = to;
    }

    /// Push a triangle to the Path.
    pub fn push_triangle(
        &mut self,
        xy: (Point<Pixels>, Point<Pixels>, Point<Pixels>),
        st: (Point<f32>, Point<f32>, Point<f32>),
    ) {
        self.bounds = self
            .bounds
            .union(&Bounds {
                origin: xy.0,
                size: Default::default(),
            })
            .union(&Bounds {
                origin: xy.1,
                size: Default::default(),
            })
            .union(&Bounds {
                origin: xy.2,
                size: Default::default(),
            });

        self.vertices.push(PathVertex {
            xy_position: xy.0,
            st_position: st.0,
            content_mask: Default::default(),
        });
        self.vertices.push(PathVertex {
            xy_position: xy.1,
            st_position: st.1,
            content_mask: Default::default(),
        });
        self.vertices.push(PathVertex {
            xy_position: xy.2,
            st_position: st.2,
            content_mask: Default::default(),
        });
    }
}

impl<T> Path<T>
where
    T: Clone + Debug + Default + PartialEq + PartialOrd + Add<T, Output = T> + Sub<Output = T>,
{
    #[allow(unused)]
    #[expect(missing_docs)]
    pub fn clipped_bounds(&self) -> Bounds<T> {
        self.bounds.intersect(&self.content_mask.bounds)
    }
}

impl From<Path<ScaledPixels>> for Primitive {
    fn from(path: Path<ScaledPixels>) -> Self {
        Primitive::Path(path)
    }
}

#[derive(Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PathVertex<P: Clone + Debug + Default + PartialEq> {
    pub xy_position: Point<P>,
    pub st_position: Point<f32>,
    pub content_mask: ContentMask<P>,
}

#[expect(missing_docs)]
impl PathVertex<Pixels> {
    pub fn scale(&self, factor: f32) -> PathVertex<ScaledPixels> {
        PathVertex {
            xy_position: self.xy_position.scale(factor),
            st_position: self.st_position,
            content_mask: self.content_mask.scale(factor),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BlurRect, LensPrimitive, LensRect, LensShape, MIRROR_AXIS_HORIZONTAL, MirrorRect,
        PrimitiveBatch, Quad, Scene,
    };
    use crate::{
        Background, BorderStyle, Bounds, ContentMask, Corners, Edges, Hsla, ScaledPixels, point,
        size,
    };

    fn test_bounds() -> Bounds<ScaledPixels> {
        Bounds {
            origin: point(ScaledPixels(10.0), ScaledPixels(20.0)),
            size: size(ScaledPixels(100.0), ScaledPixels(50.0)),
        }
    }

    fn test_content_mask() -> ContentMask<ScaledPixels> {
        ContentMask {
            bounds: Bounds {
                origin: point(ScaledPixels(0.0), ScaledPixels(0.0)),
                size: size(ScaledPixels(1000.0), ScaledPixels(1000.0)),
            },
        }
    }

    fn test_corners() -> Corners<ScaledPixels> {
        Corners {
            top_left: ScaledPixels(4.0),
            top_right: ScaledPixels(4.0),
            bottom_right: ScaledPixels(4.0),
            bottom_left: ScaledPixels(4.0),
        }
    }

    #[test]
    fn blur_rect_round_trip() {
        let mut scene = Scene::default();
        scene.insert_primitive(BlurRect {
            order: 0,
            kernel_levels: 3,
            bounds: test_bounds(),
            content_mask: test_content_mask(),
            corner_radii: test_corners(),
            blur_radius: ScaledPixels(12.0),
            blur_radius_end: ScaledPixels(12.0),
            gradient_direction: [0.0, 0.0],
            tint: Hsla {
                h: 0.0,
                s: 0.0,
                l: 0.0,
                a: 0.3,
            },
            _pad_end: [0.0; 2],
        });
        scene.finish();

        assert_eq!(scene.blur_rects.len(), 1);
        assert!(scene.lens_rects.is_empty());

        let batches: Vec<_> = scene.batches().collect();
        assert!(
            batches
                .iter()
                .any(|b| matches!(b, PrimitiveBatch::BlurRects(r) if r.len() == 1)),
            "expected a BlurRects batch of size 1, got {:?}",
            batches
        );
    }

    fn test_lens_shape() -> LensShape {
        LensShape {
            bounds: test_bounds(),
            corner_radii: test_corners(),
            _pad: [0.0; 4],
        }
    }

    fn make_lens_primitive() -> LensPrimitive {
        LensPrimitive {
            rect: LensRect {
                order: 0,
                kernel_levels: 3,
                bounds: test_bounds(),
                content_mask: test_content_mask(),
                shape_offset: 0,
                shape_count: 0,
                edge_blend_distance: ScaledPixels(0.0),
                blur_radius: ScaledPixels(8.0),
                gradient_direction: [0.0, 0.0],
                blur_radius_end: ScaledPixels(8.0),
                refraction: 12.0,
                depth: 6.0,
                dispersion: 2.0,
                splay: ScaledPixels(1.5),
                light_angle_radians: std::f32::consts::FRAC_PI_4,
                light_intensity: 0.2,
                tint: Hsla {
                    h: 0.6,
                    s: 0.2,
                    l: 0.8,
                    a: 0.1,
                },
                pad2: 0.0,
            },
            shapes: vec![test_lens_shape()],
        }
    }

    #[test]
    fn lens_rect_round_trip() {
        let mut scene = Scene::default();
        scene.insert_primitive(make_lens_primitive());
        scene.finish();

        assert_eq!(scene.lens_rects.len(), 1);
        assert_eq!(scene.lens_shapes.len(), 1);
        assert!(scene.blur_rects.is_empty());
        assert_eq!(scene.lens_rects[0].shape_offset, 0);
        assert_eq!(scene.lens_rects[0].shape_count, 1);

        let batches: Vec<_> = scene.batches().collect();
        assert!(
            batches
                .iter()
                .any(|b| matches!(b, PrimitiveBatch::LensRects(r) if r.len() == 1)),
            "expected a LensRects batch of size 1, got {:?}",
            batches
        );
    }

    #[test]
    fn lens_rect_multi_shape_roundtrip() {
        let mut scene = Scene::default();
        let mut lens = make_lens_primitive();
        lens.shapes = (0..8).map(|_| test_lens_shape()).collect();
        scene.insert_primitive(lens);
        scene.finish();

        assert_eq!(scene.lens_rects.len(), 1);
        assert_eq!(scene.lens_shapes.len(), 8);
        assert_eq!(scene.lens_rects[0].shape_offset, 0);
        assert_eq!(scene.lens_rects[0].shape_count, 8);
    }

    #[test]
    fn lens_shapes_survive_lens_rect_sort() {
        let mut scene = Scene::default();
        // Insert two lens primitives in reverse draw order; lens_shapes must
        // still be indexable by the final, sorted lens_rects.
        for (draw_order, shape_count) in [(5u32, 3usize), (1u32, 2usize)] {
            let mut lens = make_lens_primitive();
            lens.rect.order = draw_order;
            lens.shapes = (0..shape_count).map(|_| test_lens_shape()).collect();
            scene.insert_primitive(lens);
        }
        scene.finish();

        assert_eq!(scene.lens_rects.len(), 2);
        assert_eq!(scene.lens_shapes.len(), 5);
        // After sort by order, the lower-order lens (2 shapes) comes first.
        // Its shape_offset was set at insertion time and must still address
        // a valid slice of lens_shapes.
        for lens in &scene.lens_rects {
            let start = lens.shape_offset as usize;
            let end = start + lens.shape_count as usize;
            assert!(end <= scene.lens_shapes.len());
        }
    }

    fn make_blur_rect() -> BlurRect {
        BlurRect {
            order: 0,
            kernel_levels: 3,
            bounds: test_bounds(),
            content_mask: test_content_mask(),
            corner_radii: test_corners(),
            blur_radius: ScaledPixels(12.0),
            blur_radius_end: ScaledPixels(12.0),
            gradient_direction: [0.0, 0.0],
            tint: Hsla {
                h: 0.0,
                s: 0.0,
                l: 0.0,
                a: 0.3,
            },
            _pad_end: [0.0; 2],
        }
    }

    fn make_quad() -> Quad {
        Quad {
            order: 0,
            border_style: BorderStyle::default(),
            bounds: test_bounds(),
            content_mask: test_content_mask(),
            background: Background::from(Hsla {
                h: 0.0,
                s: 0.0,
                l: 0.0,
                a: 1.0,
            }),
            border_color: Hsla {
                h: 0.0,
                s: 0.0,
                l: 0.0,
                a: 0.0,
            },
            corner_radii: test_corners(),
            border_widths: Edges {
                top: ScaledPixels(0.0),
                right: ScaledPixels(0.0),
                bottom: ScaledPixels(0.0),
                left: ScaledPixels(0.0),
            },
        }
    }

    #[test]
    fn blur_rects_interleave_with_quads() {
        let mut scene = Scene::default();
        for order in 0..4 {
            if order % 2 == 0 {
                scene.insert_primitive(Quad { ..make_quad() });
            } else {
                scene.insert_primitive(BlurRect { ..make_blur_rect() });
            }
        }
        scene.finish();

        let batch_kinds: Vec<_> = scene
            .batches()
            .map(|b| match b {
                PrimitiveBatch::Quads(_) => "Quads",
                PrimitiveBatch::BlurRects(_) => "BlurRects",
                other => panic!("unexpected batch kind: {:?}", other),
            })
            .collect();
        assert_eq!(
            batch_kinds,
            vec!["Quads", "BlurRects", "Quads", "BlurRects"],
            "expected alternating Quad/BlurRects batches"
        );
    }

    #[test]
    fn max_blur_kernel_levels_clamps_to_five() {
        let mut scene = Scene::default();
        let mut blur = make_blur_rect();
        blur.kernel_levels = 10;
        scene.insert_primitive(blur);
        scene.finish();
        assert_eq!(scene.max_blur_kernel_levels(), 5);
    }

    #[test]
    fn max_blur_kernel_levels_empty_is_zero() {
        let scene = Scene::default();
        assert_eq!(scene.max_blur_kernel_levels(), 0);
    }

    #[test]
    fn max_blur_radius_returns_max_of_blur_and_lens() {
        let mut scene = Scene::default();
        let mut blur = make_blur_rect();
        blur.blur_radius = ScaledPixels(8.0);
        scene.insert_primitive(blur);

        let mut lens = make_lens_primitive();
        lens.rect.blur_radius = ScaledPixels(24.0);
        lens.rect.dispersion = 0.0;
        lens.rect.light_intensity = 0.0;
        lens.rect.light_angle_radians = 0.0;
        lens.rect.refraction = 12.0;
        lens.rect.depth = 6.0;
        lens.rect.tint = Hsla {
            h: 0.0,
            s: 0.0,
            l: 0.0,
            a: 0.0,
        };
        scene.insert_primitive(lens);
        scene.finish();

        assert_eq!(scene.max_blur_radius(), ScaledPixels(24.0));
    }

    fn make_mirror_rect() -> MirrorRect {
        MirrorRect {
            order: 0,
            kernel_levels: 3,
            bounds: test_bounds(),
            content_mask: test_content_mask(),
            corner_radii: test_corners(),
            source_bounds: test_bounds(),
            mirror_axis: MIRROR_AXIS_HORIZONTAL,
            clip_to_destination: 1,
            blur_radius: ScaledPixels(0.0),
            blur_radius_end: ScaledPixels(0.0),
            gradient_direction: [0.0, 0.0],
            tint: Hsla {
                h: 0.0,
                s: 0.0,
                l: 0.0,
                a: 0.0,
            },
        }
    }

    #[test]
    fn mirror_rect_round_trip() {
        let mut scene = Scene::default();
        scene.insert_primitive(make_mirror_rect());
        scene.finish();

        assert_eq!(scene.mirror_rects.len(), 1);

        let batches: Vec<_> = scene.batches().collect();
        assert!(
            batches
                .iter()
                .any(|b| matches!(b, PrimitiveBatch::MirrorRects(r) if r.len() == 1)),
            "expected a MirrorRects batch of size 1, got {:?}",
            batches
        );
    }

    #[test]
    fn mirror_rect_with_blur_affects_max_blur_radius() {
        let mut scene = Scene::default();
        let mut mirror = make_mirror_rect();
        mirror.blur_radius = ScaledPixels(18.0);
        mirror.blur_radius_end = ScaledPixels(30.0);
        scene.insert_primitive(mirror);
        scene.finish();
        assert_eq!(scene.max_blur_radius(), ScaledPixels(30.0));
        assert_eq!(scene.max_blur_kernel_levels(), 3);
    }

    #[test]
    fn mirror_rect_without_blur_contributes_zero_kernel() {
        let mut scene = Scene::default();
        scene.insert_primitive(make_mirror_rect());
        scene.finish();
        // blur_radius == 0 and blur_radius_end == 0: no Kawase chain needed.
        assert_eq!(scene.max_blur_kernel_levels(), 0);
    }

    #[test]
    fn blur_rects_coalesce_when_adjacent() {
        let mut scene = Scene::default();
        scene.insert_primitive(BlurRect { ..make_blur_rect() });
        scene.insert_primitive(BlurRect { ..make_blur_rect() });
        scene.finish();

        let blur_batches: Vec<_> = scene
            .batches()
            .filter_map(|b| match b {
                PrimitiveBatch::BlurRects(range) => Some(range),
                _ => None,
            })
            .collect();
        assert_eq!(
            blur_batches.len(),
            1,
            "expected a single coalesced BlurRects batch, got {:?}",
            blur_batches
        );
        assert_eq!(blur_batches[0].len(), 2);
    }
}
