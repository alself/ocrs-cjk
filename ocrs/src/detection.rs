use anyhow::anyhow;
use rten::{Dimension, FloatOperators, Operators, RunOptions};
use rten_imageproc::{find_contours, min_area_rect, simplify_polygon, RetrievalMode, RotatedRect};
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView, Tensor};

use crate::model::Model;
use crate::preprocess::BLACK_VALUE;

/// Identifies the detection model format and its expected input normalization.
enum DetectorKind {
    /// Original ocrs detection model: 1-channel greyscale CHW, fixed input
    /// dimensions, BLACK_VALUE-biased normalization.
    OcrsRten,

    /// PaddleOCR DB detection model: 3-channel RGB CHW, dynamic input
    /// dimensions (multiples of 32), ImageNet normalization.
    PaddleOcrDb,
}

/// Parameters that control post-processing of text detection model outputs.
#[derive(Clone, Debug, PartialEq)]
pub struct TextDetectorParams {
    /// Threshold for minimum area of returned rectangles.
    ///
    /// This can be used to filter out rects created by small false positives in
    /// the mask, at the risk of filtering out true positives. The more accurate
    /// the model producing the mask is, the smaller this value can be.
    pub min_area: f32,

    /// Threshold for per-pixel scores in output segmentation mask for
    /// classifying a pixel as text.
    pub text_threshold: f32,

    /// Ratio used to expand each detected rect outwards, following PaddleOCR's
    /// `DBPostProcess` "unclip" step.
    ///
    /// PaddleOCR's DB detection models are trained to predict a region that is
    /// *smaller* than the text, so the predicted box has to be grown back
    /// before it is passed to recognition. The distance applied to a rect is
    /// `area * unclip_ratio / perimeter`. PaddleOCR defaults to `1.5`.
    ///
    /// ocrs's original behaviour is a fixed `expand_dist` for every rect, which
    /// is too small for narrow boxes and truncates the leading or trailing
    /// glyph. Set this to `0.` to keep that behaviour.
    pub unclip_ratio: f32,
}

impl Default for TextDetectorParams {
    fn default() -> TextDetectorParams {
        // Empirically chosen parameters for the initial model release.
        TextDetectorParams {
            // This area is quite large and can prevent detection of small /
            // single letter words.
            min_area: 100.,

            // Ideally the threshold would be 0.5 as a neutral value.
            text_threshold: 0.2,

            // Disabled by default: the DB "shrink then unclip" convention only
            // applies to PaddleOCR detection models. ocrs's own model is not
            // trained that way.
            unclip_ratio: 0.,
        }
    }
}

/// Distance to expand `rect` by, per PaddleOCR's `DBPostProcess` unclip step:
/// `area * unclip_ratio / perimeter`.
///
/// Returns `None` when unclip is disabled (`unclip_ratio <= 0.`) or the rect is
/// degenerate, in which case the caller falls back to its fixed `expand_dist`.
fn unclip_distance(rect: &RotatedRect, unclip_ratio: f32) -> Option<f32> {
    if unclip_ratio <= 0. {
        return None;
    }
    let perimeter = 2. * (rect.width() + rect.height());
    if perimeter <= 0. {
        return None;
    }
    Some(rect.area() * unclip_ratio / perimeter)
}

/// Find the minimum-area oriented rectangles containing each connected
/// component in the binary mask `mask`.
fn find_connected_component_rects(
    mask: NdTensorView<bool, 2>,
    expand_dist: f32,
    unclip_ratio: f32,
    min_area: f32,
) -> Vec<RotatedRect> {
    find_contours(mask, RetrievalMode::External)
        .iter()
        .filter_map(|poly| {
            let float_points: Vec<_> = poly.iter().map(|p| p.to_f32()).collect();
            let simplified = simplify_polygon(&float_points, 2. /* epsilon */);

            min_area_rect(&simplified).map(|mut rect| {
                let dist = unclip_distance(&rect, unclip_ratio).unwrap_or(expand_dist);
                rect.resize(rect.width() + 2. * dist, rect.height() + 2. * dist);
                rect
            })
        })
        .filter(|r| r.area() >= min_area)
        .collect()
}

/// Text detector which finds the oriented bounding boxes of words in an input
/// image.
pub struct TextDetector {
    model: Box<dyn Model + Send + Sync>,
    params: TextDetectorParams,
    input_shape: Vec<Dimension>,
    kind: DetectorKind,
}

impl TextDetector {
    /// Initialize a DetectionModel from a trained RTen model.
    ///
    /// This will fail if the model doesn't have the expected inputs or outputs.
    pub fn from_model(
        model: impl Model + Send + Sync + 'static,
        params: TextDetectorParams,
    ) -> anyhow::Result<TextDetector> {
        let input_shape = model.input_shape()?;

        // Auto-detect model kind from the channel dimension (index 1 in NCHW).
        // PaddleOCR DB detection models expect 3-channel RGB input;
        // the original ocrs model expects 1-channel greyscale.
        let kind = match input_shape.get(1) {
            Some(Dimension::Fixed(3)) => DetectorKind::PaddleOcrDb,
            _ => DetectorKind::OcrsRten,
        };

        Ok(TextDetector {
            model: Box::new(model),
            params,
            input_shape,
            kind,
        })
    }

    /// Return true if this detector requires a 3-channel RGB input image
    /// (prepared with ImageNet normalization) rather than the standard
    /// 1-channel greyscale input.
    pub(crate) fn needs_color_image(&self) -> bool {
        matches!(self.kind, DetectorKind::PaddleOcrDb)
    }

    /// Return the confidence threshold used to determine whether a pixel is
    /// text or not.
    pub fn threshold(&self) -> f32 {
        self.params.text_threshold
    }

    /// Detect text words in a greyscale image.
    ///
    /// `image` is a greyscale CHW image with values in the range `ZERO_VALUE` to
    /// `ZERO_VALUE + 1`. `model` is a model which takes an NCHW input tensor and
    /// returns a binary segmentation mask predicting whether each pixel is part of
    /// a text word or not. The image is padded and resized to the model's expected
    /// input size before performing detection.
    ///
    /// The result is an unsorted list of the oriented bounding rectangles of
    /// connected components (ie. text words) in the mask.
    pub fn detect_words(
        &self,
        image: NdTensorView<f32, 3>,
        debug: bool,
    ) -> anyhow::Result<Vec<RotatedRect>> {
        let text_mask = self.detect_text_pixels(image, debug)?;
        let binary_mask = text_mask.map(|prob| *prob > self.params.text_threshold);

        // Distance to expand bounding boxes by. This is useful when the model is
        // trained to assign a positive label to pixels in a smaller area than the
        // ground truth, which may be done to create separation between adjacent
        // objects. Used as-is unless `unclip_ratio` overrides it per rect.
        let expand_dist = 3.;

        let word_rects = find_connected_component_rects(
            binary_mask.view(),
            expand_dist,
            self.params.unclip_ratio,
            self.params.min_area,
        );

        Ok(word_rects)
    }

    /// Detect text pixels in an image.
    ///
    /// Takes a CHW input image and returns a probability map indicating whether
    /// each pixel in the input is text. For [DetectorKind::OcrsRten] the input
    /// must be a 1-channel greyscale image; for [DetectorKind::PaddleOcrDb]
    /// the input must be a 3-channel ImageNet-normalised RGB image.
    ///
    /// See [detect_words](TextDetector::detect_words) for more details of
    /// expected input.
    pub fn detect_text_pixels(
        &self,
        image: NdTensorView<f32, 3>,
        debug: bool,
    ) -> anyhow::Result<NdTensor<f32, 2>> {
        let [img_chans, img_height, img_width] = image.shape();

        // Add batch dim
        let image = image.reshaped([1, img_chans, img_height, img_width]);

        // nb. RunOptions will be non_exhaustive in future.
        #[allow(clippy::field_reassign_with_default)]
        let opts = {
            let mut opts = RunOptions::default();
            opts.timing = debug;
            opts
        };

        match self.kind {
            DetectorKind::OcrsRten => {
                let [_, _, Dimension::Fixed(in_height), Dimension::Fixed(in_width)] =
                    self.input_shape[..]
                else {
                    return Err(anyhow!("failed to get model dims"));
                };

                // Pad small images to the input size of the text detection model.
                let pad_bottom = (in_height as i32 - img_height as i32).max(0);
                let pad_right = (in_width as i32 - img_width as i32).max(0);
                let image = (pad_bottom > 0 || pad_right > 0)
                    .then(|| {
                        let pads = &[0, 0, 0, 0, 0, 0, pad_bottom, pad_right];
                        image.pad(pads.into(), BLACK_VALUE)
                    })
                    .transpose()?
                    .map(|t| t.into_cow())
                    .unwrap_or(image.as_dyn().as_cow());

                // Resize images to the text detection model's input size.
                let image = (image.size(2) != in_height || image.size(3) != in_width)
                    .then(|| image.resize_image([in_height, in_width]))
                    .transpose()?
                    .map(|t| t.into_cow())
                    .unwrap_or(image);

                let text_mask: Tensor<f32> = self.model.run(image.view(), Some(opts))?;

                // Resize probability mask to original input size, cropping out
                // the padding region first.
                let text_mask = text_mask
                    .slice((
                        ..,
                        ..,
                        ..(in_height - pad_bottom as usize),
                        ..(in_width - pad_right as usize),
                    ))
                    .resize_image([img_height, img_width])?;

                // Remove batch, channel dims.
                Ok(text_mask.into_shape([img_height, img_width]))
            }

            DetectorKind::PaddleOcrDb => {
                // Resize to nearest multiple of 32, capping the longer side at
                // 960 px to keep memory usage bounded.
                const MAX_SIDE: usize = 960;
                const STRIDE: usize = 32;

                let long_side = img_height.max(img_width);
                let scale = if long_side > MAX_SIDE {
                    MAX_SIDE as f32 / long_side as f32
                } else {
                    1.0_f32
                };

                let round32 = |v: usize| -> usize {
                    (((v as f32 * scale) / STRIDE as f32).round() as usize * STRIDE).max(STRIDE)
                };
                let new_h = round32(img_height);
                let new_w = round32(img_width);

                let image = (new_h != img_height || new_w != img_width)
                    .then(|| image.resize_image([new_h, new_w]))
                    .transpose()?
                    .map(|t| t.into_cow())
                    .unwrap_or(image.as_dyn().as_cow());

                let text_mask: Tensor<f32> = self.model.run(image.view(), Some(opts))?;

                // Resize probability mask back to original image dimensions.
                let text_mask = text_mask.resize_image([img_height, img_width])?;

                // Remove batch, channel dims.
                Ok(text_mask.into_shape([img_height, img_width]))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use rten_imageproc::{fill_rect, Point, RectF, RotatedRect};
    use rten_tensor::prelude::*;
    use rten_tensor::NdTensor;

    use super::{find_connected_component_rects, unclip_distance};
    use crate::test_util::gen_rect_grid;

    #[test]
    fn test_unclip_distance() {
        // PaddleOCR's DBPostProcess: distance = area * unclip_ratio / perimeter.
        // For a 50x10 rect: 500 * 1.5 / (2 * (50 + 10)) = 6.25.
        let rect = RotatedRect::from_rect(RectF::from_tlbr(0., 0., 10., 50.));
        assert_eq!(unclip_distance(&rect, 1.5), Some(6.25));

        // Unclip disabled: the caller falls back to its fixed expand distance.
        assert_eq!(unclip_distance(&rect, 0.), None);
        assert_eq!(unclip_distance(&rect, -1.), None);

        // Degenerate rect has no perimeter to divide by.
        let flat = RotatedRect::from_rect(RectF::from_tlbr(0., 0., 0., 0.));
        assert_eq!(unclip_distance(&flat, 1.5), None);
    }

    #[test]
    fn test_find_connected_component_rects_unclip() {
        let (rect_h, rect_w) = (10, 50);
        let mut mask = NdTensor::zeros([100, 100]);
        let rects = gen_rect_grid(
            Point::from_yx(10, 10),
            (1, 1),           /* grid_shape */
            (rect_h, rect_w), /* rect_size */
            (10, 5),          /* gap_size */
        );
        for r in rects.iter() {
            fill_rect(mask.view_mut(), r.adjust_tlbr(0, 0, 1, 1), true);
        }

        let min_area = 100.;
        let plain = find_connected_component_rects(mask.view(), 0., 0., min_area);
        let unclipped = find_connected_component_rects(mask.view(), 0., 1.5, min_area);
        assert_eq!(plain.len(), 1);
        assert_eq!(unclipped.len(), 1);

        // A 10x50 mask component is expanded by 6.25 on each side.
        let expected = (rect_w as f32 + 2. * 6.25, rect_h as f32 + 2. * 6.25);
        assert!(
            (unclipped[0].width() - expected.0).abs() < 1.,
            "width {} != expected {}",
            unclipped[0].width(),
            expected.0
        );
        assert!(
            (unclipped[0].height() - expected.1).abs() < 1.,
            "height {} != expected {}",
            unclipped[0].height(),
            expected.1
        );
    }

    #[test]
    fn test_find_connected_component_rects() {
        let mut mask = NdTensor::zeros([400, 400]);
        let (grid_h, grid_w) = (5, 5);
        let (rect_h, rect_w) = (10, 50);
        let rects = gen_rect_grid(
            Point::from_yx(10, 10),
            (grid_h, grid_w), /* grid_shape */
            (rect_h, rect_w), /* rect_size */
            (10, 5),          /* gap_size */
        );
        for r in rects.iter() {
            // Expand `r` because `fill_rect` does not set points along the
            // right/bottom boundary.
            let expanded = r.adjust_tlbr(0, 0, 1, 1);
            fill_rect(mask.view_mut(), expanded, true);
        }

        let min_area = 100.;
        let components = find_connected_component_rects(mask.view(), 0., 0., min_area);
        assert_eq!(components.len() as i32, grid_h * grid_w);

        for c in components.iter() {
            let mut shape = [c.height().round() as i32, c.width().round() as i32];
            shape.sort();

            // We sort the dimensions before comparison here to be invariant to
            // different rotations of the connected component that cover the
            // same pixels.
            let mut expected_shape = [rect_h, rect_w];
            expected_shape.sort();

            assert_eq!(shape, expected_shape);
        }
    }
}
