use std::collections::HashSet;

use anyhow::anyhow;
use rten_imageproc::RotatedRect;
use rten_tensor::prelude::*;
use rten_tensor::NdTensor;

mod detection;
mod errors;
mod geom_util;
mod layout_analysis;
mod log;
mod model;
mod preprocess;
mod recognition;

#[cfg(test)]
mod test_util;

mod text_items;

#[cfg(all(feature = "export-wasm", target_arch = "wasm32"))]
mod wasm_api;

pub mod cjk_text;
pub use cjk_text::{cjk_alphabet, cjk_alphabet_chars, is_cjk, segment};

use detection::TextDetector;
use layout_analysis::find_text_lines;
use model::Model;
use preprocess::{prepare_image, prepare_image_imagenet, BLACK_VALUE as PREPROCESS_BLACK_VALUE};
use recognition::{RecognitionOpt, TextRecognizer};

pub use detection::TextDetectorParams;
pub use preprocess::{DimOrder, ImagePixels, ImageSource, ImageSourceError};
pub use recognition::DecodeMethod;
pub use text_items::{TextChar, TextItem, TextLine, TextWord};

const DEFAULT_ALPHABET: &str = " 0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~€ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

/// Configuration for an [OcrEngine] instance.
#[derive(Default)]
pub struct OcrEngineParams {
    /// Model used to detect text words in the image.
    pub detection_model: Option<rten::Model>,

    /// Model used to recognize lines of text in the image.
    ///
    /// If using a custom model, you may need to adjust the
    /// [`alphabet`](Self::alphabet) to match.
    pub recognition_model: Option<rten::Model>,

    /// Enable debug logging.
    pub debug: bool,

    /// Recognize each rectangle returned by the detection model independently
    /// instead of first merging them into lines with layout analysis.
    ///
    /// This matches the PaddleOCR / RapidOCR pipeline, which runs recognition
    /// once per detection output box. It is the correct granularity when the
    /// detection model already emits one box per text line or per table cell
    /// (eg. PP-OCR DB models): ocrs's own layout analysis would otherwise merge
    /// boxes that share a visual row but belong to different table columns,
    /// which inflates the crop passed to the recognition model and destroys
    /// small text.
    pub rec_per_detected_rect: bool,

    /// Parameters controlling post-processing of the detection model's output.
    ///
    /// Set [`TextDetectorParams::unclip_ratio`] when using a PaddleOCR DB
    /// detection model, which predicts boxes that are shrunk relative to the
    /// text and must be expanded back before recognition.
    pub detection_params: TextDetectorParams,

    /// Method used to decode outputs of text recognition model.
    pub decode_method: DecodeMethod,

    /// Alphabet used for text recognition.
    ///
    /// This is useful if you are using a custom recognition model with a
    /// modified alphabet. If not specified a default alphabet will be used
    /// which matches the one used to train the [original
    /// models](https://github.com/robertknight/ocrs-models).
    pub alphabet: Option<String>,

    /// Alphabet as `Vec<char>`, bypassing the UTF-8 round-trip of [`alphabet`](Self::alphabet).
    ///
    /// Takes precedence over `alphabet` when both are set. Prefer this when you
    /// already have a `Vec<char>`, for example from [`cjk_text::cjk_alphabet_chars()`]:
    ///
    /// ```ignore
    /// OcrEngineParams {
    ///     alphabet_chars: Some(cjk_text::cjk_alphabet_chars()),
    ///     ..Default::default()
    /// }
    /// ```
    pub alphabet_chars: Option<Vec<char>>,

    /// Set of characters that may be produced by text recognition.
    ///
    /// This is useful when you need the text recognition model to
    /// produce text that only includes a predefined set of characters, for
    /// example only numbers or lower-case letters.
    ///
    /// If this option is not set, text recognition may produce any character
    /// from the recognition model's alphabet.
    pub allowed_chars: Option<String>,
}

/// Internal counterpart to [`OcrEngineParams`] which is generic over the
/// inference engine.
#[derive(Default)]
struct OcrEngineParamsImpl<M: Model> {
    detection_model: Option<M>,
    recognition_model: Option<M>,
    debug: bool,
    decode_method: DecodeMethod,
    alphabet: Option<String>,
    alphabet_chars: Option<Vec<char>>,
    allowed_chars: Option<String>,
    rec_per_detected_rect: bool,
    detection_params: TextDetectorParams,
}

impl From<OcrEngineParams> for OcrEngineParamsImpl<rten::Model> {
    fn from(params: OcrEngineParams) -> Self {
        let OcrEngineParams {
            detection_model,
            recognition_model,
            debug,
            decode_method,
            alphabet,
            alphabet_chars,
            allowed_chars,
            rec_per_detected_rect,
            detection_params,
        } = params;

        Self {
            detection_model,
            recognition_model,
            debug,
            decode_method,
            alphabet,
            alphabet_chars,
            allowed_chars,
            rec_per_detected_rect,
            detection_params,
        }
    }
}

/// Detects and recognizes text in images.
///
/// OcrEngine uses machine learning models to detect text, analyze layout
/// and recognize text in an image.
pub struct OcrEngine {
    detector: Option<TextDetector>,
    recognizer: Option<TextRecognizer>,
    debug: bool,
    decode_method: DecodeMethod,
    alphabet_chars: Vec<char>,

    /// When true, recognition runs once per detected rect rather than once per
    /// line produced by layout analysis. See
    /// [`OcrEngineParams::rec_per_detected_rect`].
    rec_per_detected_rect: bool,

    /// Indices of characters in `alphabet` that are excluded from recognition
    /// output. See [`OcrEngineParams::allowed_chars`].
    excluded_char_labels: Option<Vec<usize>>,
}

/// Input image for OCR analysis. Instances are created using
/// [OcrEngine::prepare_input]
pub struct OcrInput {
    /// 1-channel greyscale CHW tensor with values in [BLACK_VALUE, BLACK_VALUE + 1.].
    pub(crate) image: NdTensor<f32, 3>,

    /// 3-channel RGB CHW tensor with ImageNet normalization.
    /// Only populated when a PaddleOCR DB detection model is loaded.
    pub(crate) color_image: Option<NdTensor<f32, 3>>,
}

impl OcrEngine {
    /// Construct a new engine from a given configuration.
    pub fn new(params: OcrEngineParams) -> anyhow::Result<OcrEngine> {
        Self::new_impl(params.into())
    }

    /// Internal constructor which allows using a dummy inference engine in tests.
    fn new_impl<M: Model + Send + Sync + 'static>(
        params: OcrEngineParamsImpl<M>,
    ) -> anyhow::Result<OcrEngine> {
        // Copy `detection_params` out first: `detection_model` is moved by the
        // `map` below and the closure cannot also move the params field.
        let detection_params = params.detection_params.clone();
        let detector = params
            .detection_model
            .map(|model| TextDetector::from_model(model, detection_params))
            .transpose()?;
        let recognizer = params
            .recognition_model
            .map(TextRecognizer::from_model)
            .transpose()?;

        let alphabet_chars: Vec<char> = match params.alphabet_chars {
            Some(chars) => chars,
            None => match params.alphabet {
                Some(s) => s.chars().collect(),
                None => DEFAULT_ALPHABET.chars().collect(),
            },
        };

        let excluded_char_labels = params.allowed_chars.map(|allowed_characters| {
            // Build a set for O(1) membership check; avoids O(alphabet × allowed) comparisons.
            let allowed_set: HashSet<char> = allowed_characters.chars().collect();
            // Index `0` is reserved for the CTC blank character and
            // `i + 1` is used as training label for character at
            // index `i` of `alphabet` string.
            //
            // See https://github.com/robertknight/ocrs-models/blob/3d98fc655d6fd4acddc06e7f5d60a55b55748a48/ocrs_models/datasets/util.py#L113
            alphabet_chars
                .iter()
                .enumerate()
                .filter_map(|(index, &char)| {
                    if !allowed_set.contains(&char) {
                        Some(index + 1)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        });

        Ok(OcrEngine {
            detector,
            recognizer,
            alphabet_chars,
            excluded_char_labels,
            debug: params.debug,
            decode_method: params.decode_method,
            rec_per_detected_rect: params.rec_per_detected_rect,
        })
    }

    /// Preprocess an image for use with other methods of the engine.
    pub fn prepare_input(&self, image: ImageSource) -> anyhow::Result<OcrInput> {
        let needs_color = self
            .detector
            .as_ref()
            .map(|d| d.needs_color_image())
            .unwrap_or(false);

        if needs_color {
            // Produce the 3-channel ImageNet-normalised tensor first,
            // then derive greyscale from it to avoid consuming `image` twice.
            let color_img = prepare_image_imagenet(image);
            let [_, height, width] = color_img.shape();

            // ITU BT.601 weights for RGB → luminance (same as `prepare_image`).
            const W: [f32; 3] = [0.299, 0.587, 0.114];
            // ImageNet normalisation constants (must mirror those in preprocess.rs).
            const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
            const STD: [f32; 3] = [0.229, 0.224, 0.225];

            let mut grey = NdTensor::zeros([1, height, width]);
            for y in 0..height {
                for x in 0..width {
                    // Undo ImageNet normalisation to recover the [0, 1] pixel value.
                    let unit = |c: usize| (color_img[[c, y, x]] * STD[c] + MEAN[c]).clamp(0.0, 1.0);
                    let g = W[0] * unit(0) + W[1] * unit(1) + W[2] * unit(2);
                    grey[[0, y, x]] = PREPROCESS_BLACK_VALUE + g;
                }
            }

            Ok(OcrInput {
                image: grey,
                color_image: Some(color_img),
            })
        } else {
            Ok(OcrInput {
                image: prepare_image(image),
                color_image: None,
            })
        }
    }

    /// Detect text words in an image.
    ///
    /// Returns an unordered list of the oriented bounding rectangles of each
    /// word found.
    pub fn detect_words(&self, input: &OcrInput) -> anyhow::Result<Vec<RotatedRect>> {
        if let Some(detector) = self.detector.as_ref() {
            let detect_view = if detector.needs_color_image() {
                input
                    .color_image
                    .as_ref()
                    .map(|c| c.view())
                    .unwrap_or(input.image.view())
            } else {
                input.image.view()
            };
            detector.detect_words(detect_view, self.debug)
        } else {
            Err(anyhow!("Detection model not loaded"))
        }
    }

    /// Detect text pixels in an image.
    ///
    /// Returns an (H, W) tensor indicating the probability of each pixel in the
    /// input being part of a text word. This is a low-level API that is useful
    /// for debugging purposes. Use [detect_words](OcrEngine::detect_words) for
    /// a higher-level API that returns oriented bounding boxes of words.
    pub fn detect_text_pixels(&self, input: &OcrInput) -> anyhow::Result<NdTensor<f32, 2>> {
        if let Some(detector) = self.detector.as_ref() {
            let detect_view = if detector.needs_color_image() {
                input
                    .color_image
                    .as_ref()
                    .map(|c| c.view())
                    .unwrap_or(input.image.view())
            } else {
                input.image.view()
            };
            detector.detect_text_pixels(detect_view, self.debug)
        } else {
            Err(anyhow!("Detection model not loaded"))
        }
    }

    /// Perform layout analysis to group words into lines and sort them in
    /// reading order.
    ///
    /// `words` is an unordered list of text word rectangles found by
    /// [OcrEngine::detect_words]. The result is a list of lines, in reading
    /// order. Each line is a sequence of word bounding rectangles, in reading
    /// order.
    pub fn find_text_lines(
        &self,
        _input: &OcrInput,
        words: &[RotatedRect],
    ) -> Vec<Vec<RotatedRect>> {
        let lines = find_text_lines(words);
        if self.rec_per_detected_rect {
            // Keep the reading order that layout analysis computed, but make
            // each detected rect its own line so that recognition runs once per
            // rect (matching PaddleOCR / RapidOCR) instead of once per merged
            // line.
            lines.into_iter().flatten().map(|rect| vec![rect]).collect()
        } else {
            lines
        }
    }

    /// Recognize lines of text in an image.
    ///
    /// `lines` is an ordered list of the text line boxes in an image,
    /// produced by [OcrEngine::find_text_lines].
    ///
    /// The output is a list of [TextLine]s corresponding to the input image
    /// regions. Entries can be `None` if no text was found in a given line.
    pub fn recognize_text(
        &self,
        input: &OcrInput,
        lines: &[Vec<RotatedRect>],
    ) -> anyhow::Result<Vec<Option<TextLine>>> {
        if let Some(recognizer) = self.recognizer.as_ref() {
            recognizer.recognize_text_lines(
                input.image.view(),
                lines,
                RecognitionOpt {
                    debug: self.debug,
                    decode_method: self.decode_method,
                    alphabet_chars: &self.alphabet_chars,
                    excluded_char_labels: self.excluded_char_labels.as_deref(),
                },
            )
        } else {
            Err(anyhow!("Recognition model not loaded"))
        }
    }

    /// Prepare an image for input into the text line recognition model.
    ///
    /// This method exists to help with debugging recognition issues by exposing
    /// the preprocessing that [OcrEngine::recognize_text] does before it feeds
    /// an image into the recognition model. Use [OcrEngine::recognize_text] to
    /// recognize text.
    ///
    /// `line` is a sequence of [RotatedRect]s that make up a line of text.
    ///
    /// Returns a greyscale (H, W) image with values in [-0.5, 0.5].
    pub fn prepare_recognition_input(
        &self,
        input: &OcrInput,
        line: &[RotatedRect],
    ) -> anyhow::Result<NdTensor<f32, 2>> {
        let Some(recognizer) = self.recognizer.as_ref() else {
            return Err(anyhow!("Recognition model not loaded"));
        };
        let line_image = recognizer.prepare_input(input.image.view(), line);
        Ok(line_image)
    }

    /// Return the alphabet the engine was configured with, as a slice of chars.
    pub fn alphabet(&self) -> &[char] {
        &self.alphabet_chars
    }

    /// Return the confidence threshold applied to the output of the text
    /// detection model to determine whether a pixel is text or not.
    pub fn detection_threshold(&self) -> f32 {
        self.detector
            .as_ref()
            .map(|detector| detector.threshold())
            .unwrap_or(TextDetectorParams::default().text_threshold)
    }

    /// Convenience API that extracts all text from an image as a single string.
    pub fn get_text(&self, input: &OcrInput) -> anyhow::Result<String> {
        use std::fmt::Write;
        let word_rects = self.detect_words(input)?;
        let line_rects = self.find_text_lines(input, &word_rects);
        let lines = self.recognize_text(input, &line_rects)?;
        let mut out = String::new();
        let mut first = true;
        for line in lines.into_iter().flatten() {
            if !first {
                out.push('\n');
            }
            write!(out, "{}", line).unwrap();
            first = false;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use rten::Dimension;
    use rten_imageproc::{fill_rect, BoundingRect, Rect, RectF, RotatedRect};
    use rten_tensor::prelude::*;
    use rten_tensor::TensorView;
    use rten_tensor::{NdTensor, NdTensorView, Tensor};

    use super::{DimOrder, ImageSource, Model, OcrEngine, OcrEngineParamsImpl, DEFAULT_ALPHABET};

    /// Generate a dummy CHW input image for OCR processing.
    ///
    /// The result is an RGB image which is black except for one line containing
    /// `n_words` white-filled rects.
    fn gen_test_image(n_words: usize) -> NdTensor<f32, 3> {
        let mut image = NdTensor::zeros([3, 100, 200]);

        for word_idx in 0..n_words {
            for chan_idx in 0..3 {
                fill_rect(
                    image.slice_mut([chan_idx]),
                    Rect::from_tlhw(30, (word_idx * 70) as i32, 20, 50),
                    1.,
                );
            }
        }

        image
    }

    /// Fake text detection model.
    ///
    /// Takes a CHW input tensor with values in `[-0.5, 0.5]` and adds a +0.5
    /// bias to produce an output "probability map".
    #[derive(Default)]
    struct FakeDetectionModel {}

    impl Model for FakeDetectionModel {
        fn input_shape(&self) -> anyhow::Result<Vec<Dimension>> {
            Ok([
                Dimension::Symbolic("batch".to_string()),
                Dimension::Fixed(1),
                // The real model uses larger inputs (800x600). The fake uses
                // smaller inputs to make tests run faster.
                Dimension::Fixed(200),
                Dimension::Fixed(100),
            ]
            .into())
        }

        fn run(
            &self,
            input: TensorView<f32>,
            _opts: Option<rten::RunOptions>,
        ) -> anyhow::Result<Tensor<f32>> {
            Ok(input.map(|v| v + 0.5))
        }
    }

    /// Fake text recognition model.
    ///
    /// This takes an NCHW input with C=1, H=64 and returns an output with
    /// shape `[W / 4, N, C]`. In the real model the last dimension is the
    /// log-probability of each class label. In this fake we just re-interpret
    /// each column of the input as a vector of probabilities.
    ///
    /// Returns a `(model, alphabet)` tuple.
    #[derive(Default)]
    struct FakeRecognitionModel {}

    impl Model for FakeRecognitionModel {
        fn input_shape(&self) -> anyhow::Result<Vec<Dimension>> {
            let output_columns = 64;
            Ok([
                Dimension::Symbolic("batch".to_string()),
                Dimension::Fixed(1),
                Dimension::Fixed(output_columns),
                Dimension::Symbolic("seq".to_string()),
            ]
            .into())
        }

        fn run(
            &self,
            input: TensorView<f32>,
            _opts: Option<rten::RunOptions>,
        ) -> anyhow::Result<Tensor<f32>> {
            let nchw: NdTensorView<f32, 4> = input.try_into()?;
            assert_eq!(nchw.size(1), 1);

            let nhw = nchw.slice((.., 0)); // Remove channel axis
            let [batch, height, width] = nhw.shape();
            assert_eq!(height, 64);

            // Width downsampling factor.
            const W_SCALE: usize = 4;

            // Max-pool to reduce width scale.
            let mut output = NdTensor::zeros([batch, height, width / W_SCALE]);
            for n in 0..batch {
                for h in 0..height {
                    for w_block in 0..width / W_SCALE {
                        let mut max = f32::MIN;
                        for i in 0..W_SCALE {
                            max = max.max(nhw[[n, h, w_block * W_SCALE + i]]);
                        }
                        output[[n, h, w_block]] = max;
                    }
                }
            }

            // Transpose NHW/4 => W/4NH
            output.permute([2, 0, 1]);
            output.make_contiguous();

            Ok(output.into())
        }
    }

    fn make_alphabet() -> String {
        let output_columns = 64;
        DEFAULT_ALPHABET.chars().take(output_columns - 1).collect()
    }

    /// Return expected word locations for an image generated by
    /// `gen_test_image(3)`.
    ///
    /// The output boxes are slightly larger than in input image. This is
    /// because the real detection model is trained to predict boxes that are
    /// slightly smaller than the ground truth, in order to create a gap between
    /// adjacent boxes. The connected components in model outputs are then
    /// expanded in post-processing to recover the correct boxes.
    fn expected_word_boxes() -> Vec<RectF> {
        let [top, height] = [27, 25];
        [
            Rect::from_tlhw(top, -3, height, 56).to_f32(),
            Rect::from_tlhw(top, 66, height, 57).to_f32(),
            Rect::from_tlhw(top, 136, height, 57).to_f32(),
        ]
        .into()
    }

    #[test]
    fn test_ocr_engine_prepare_input() -> Result<(), Box<dyn Error>> {
        let image = gen_test_image(3 /* n_words */);
        let engine = OcrEngine::new_impl(OcrEngineParamsImpl {
            detection_model: Some(FakeDetectionModel {}),
            recognition_model: None,
            ..Default::default()
        })?;
        let input = engine.prepare_input(ImageSource::from_tensor(image.view(), DimOrder::Chw)?)?;

        let [chans, height, width] = input.image.shape();
        assert_eq!(chans, 1);
        assert_eq!(width, image.size(2));
        assert_eq!(height, image.size(1));

        Ok(())
    }

    #[test]
    fn test_ocr_engine_detect_words() -> Result<(), Box<dyn Error>> {
        let n_words = 3;
        let image = gen_test_image(n_words);
        let engine = OcrEngine::new_impl(OcrEngineParamsImpl {
            detection_model: Some(FakeDetectionModel {}),
            recognition_model: None,
            ..Default::default()
        })?;
        let input = engine.prepare_input(ImageSource::from_tensor(image.view(), DimOrder::Chw)?)?;
        let words = engine.detect_words(&input)?;

        assert_eq!(words.len(), n_words);

        let mut boxes: Vec<RectF> = words
            .into_iter()
            .map(|rotated_rect| rotated_rect.bounding_rect())
            .collect();
        boxes.sort_by_key(|b| [b.top() as i32, b.left() as i32]);

        assert_eq!(boxes, expected_word_boxes());

        Ok(())
    }

    // Test recognition using a dummy recognition model.
    //
    // The dummy model treats each column of the input image as a vector of
    // character class probabilities. Pre-processing of the input will shift
    // values from [0, 1] to [-0.5, 0.5]. CTC decoding of the output will ignore
    // class 0 (as it represents a CTC blank) and repeated characters.
    //
    // Filling a single input row with "1"s will produce a single char output
    // where the char's index in the alphabet is the row index - 1.  ie. Filling
    // the first row produces " ", the second row "0" and so on, using the
    // default alphabet.
    fn test_recognition(
        engine: OcrEngine,
        image: NdTensorView<f32, 3>,
        expected_text: &str,
    ) -> Result<(), Box<dyn Error>> {
        let input = engine.prepare_input(ImageSource::from_tensor(image.view(), DimOrder::Chw)?)?;

        // Create a dummy input line with a single word which fills the image.
        let mut line_regions: Vec<Vec<RotatedRect>> = Vec::new();
        line_regions.push(
            [Rect::from_tlhw(0, 0, image.shape()[1] as i32, image.shape()[2] as i32).to_f32()]
                .map(RotatedRect::from_rect)
                .into(),
        );

        let lines = engine.recognize_text(&input, &line_regions)?;
        assert_eq!(lines.len(), line_regions.len());

        assert!(lines.get(0).is_some());
        let line = lines[0].as_ref().unwrap();
        assert_eq!(line.to_string(), expected_text);

        Ok(())
    }

    #[test]
    fn test_ocr_engine_recognize_lines() -> Result<(), Box<dyn Error>> {
        let mut image = NdTensor::zeros([1, 64, 32]);

        // Set the probability of character 1 in the alphabet ('0') to 1 and
        // leave all other characters with a probability of zero.
        image.slice_mut((.., 2, ..)).fill(1.);

        let rec_model = FakeRecognitionModel {};
        let engine = OcrEngine::new_impl(OcrEngineParamsImpl {
            detection_model: None,
            recognition_model: Some(rec_model),
            alphabet: Some(make_alphabet()),
            ..Default::default()
        })?;
        test_recognition(engine, image.view(), "0")?;

        Ok(())
    }

    #[test]
    fn test_ocr_engine_filter_chars() -> Result<(), Box<dyn Error>> {
        let mut image = NdTensor::zeros([1, 64, 32]);

        // Set the probability of "0" to 0.7 and "1" to 0.3.
        image.slice_mut((.., 2, ..)).fill(0.7);
        image.slice_mut((.., 3, ..)).fill(0.3);

        let alphabet = make_alphabet();

        let rec_model = FakeRecognitionModel {};
        let engine = OcrEngine::new_impl(OcrEngineParamsImpl {
            detection_model: None,
            recognition_model: Some(rec_model),
            alphabet: Some(alphabet.clone()),
            ..Default::default()
        })?;
        test_recognition(engine, image.view(), "0")?;

        // Run recognition again but exclude "0" from the output.
        let rec_model = FakeRecognitionModel {};
        let engine = OcrEngine::new_impl(OcrEngineParamsImpl {
            detection_model: None,
            recognition_model: Some(rec_model),
            alphabet: Some(alphabet),
            allowed_chars: Some("123456789".into()),
            ..Default::default()
        })?;
        test_recognition(engine, image.view(), "1")?;

        Ok(())
    }
}
